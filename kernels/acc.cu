#include "acc.hpp"
#include "cuda_check.hpp"
#include <cstdio>
#include <algorithm>

// ============================================================
// LINE-ENHANCE kernel (Chu et al., 2024) -- unchanged from Deliverable 1
// ============================================================

__inline__ __device__ float warp_prefix_sum(float val, int lane, int width)
{
    for (int offset = 1; offset < width; offset <<= 1)
    {
        float n = __shfl_up_sync(0xffffffff, val, offset, width);
        if (lane >= offset)
            val += n;
    }
    return val;
}

template <int R, int V>
__global__ void spmv_line_enhance(int nrows,
                                  int rows_per_wg,
                                  const int *__restrict__ row_ptr,
                                  const int *__restrict__ col_idx,
                                  const float *__restrict__ values,
                                  const float *__restrict__ x,
                                  float *__restrict__ y)
{
    extern __shared__ float shared_mem[]; // size = R * THREADS floats

    const int THREADS = blockDim.x;
    const int N_SHARED = R * THREADS;

    const int tid = threadIdx.x;
    const int wg_id = blockIdx.x;

    const int row_begin = wg_id * rows_per_wg;
    const int row_end = min(row_begin + rows_per_wg, nrows);

    if (row_begin >= nrows)
        return;

    const int nnz_start = row_ptr[row_begin];
    const int nnz_end = row_ptr[row_end];
    const int local_nnz = nnz_end - nnz_start;

    const int rounds = (local_nnz + N_SHARED - 1) / N_SHARED;

    const int vec_i = tid / V;
    const int tid_in_vec = tid % V;

    float sum = 0.0f;

    for (int r = 0; r < rounds; ++r)
    {
        const int round_inx_start = nnz_start + r * N_SHARED;
        const int round_inx_end = min(round_inx_start + N_SHARED, nnz_end);

        int i = round_inx_start + tid;
#pragma unroll
        for (int k = 0; k < R; ++k)
        {
            int shared_slot = tid + k * THREADS;
            if (i < round_inx_end)
                shared_mem[shared_slot] = values[i] * x[col_idx[i]];
            else
                shared_mem[shared_slot] = 0.0f;
            i += THREADS;
        }

        __syncthreads();

        int reduce_row_id = row_begin + vec_i;
        float local_sum = 0.0f;

        if (reduce_row_id < row_end)
        {
            int Ibegin = row_ptr[reduce_row_id];
            int Iend = row_ptr[reduce_row_id + 1];

            int Rbegin = max(Ibegin, round_inx_start);
            int Rend = min(Iend, round_inx_end);

            int sm_base = round_inx_start - nnz_start;

            for (int j = Rbegin + tid_in_vec; j < Rend; j += V)
                local_sum += shared_mem[j - nnz_start - sm_base];
        }

        if (V > 1)
        {
            local_sum = warp_prefix_sum(local_sum, tid_in_vec, V);
            if (tid_in_vec == V - 1 && reduce_row_id < row_end)
                sum += local_sum;
        }
        else
        {
            if (reduce_row_id < row_end)
                sum += local_sum;
        }

        __syncthreads();
    }

    bool is_leader = (V == 1) ? true : (tid_in_vec == V - 1);

    int reduce_row_id = row_begin + vec_i;
    if (is_leader && reduce_row_id < row_end)
        y[reduce_row_id] = sum;
}

// ============================================================
// FLAT kernel (Chu et al., 2024) -- unchanged from Deliverable 1
// ============================================================

__global__ void flat_preprocess(int m,
                                const int *__restrict__ row_ptr,
                                int *__restrict__ bp,
                                int STRIDE,
                                int WGS)
{
    int total = gridDim.x * blockDim.x;
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (g_tid == 0)
        bp[0] = 0;

    for (int i = g_tid; i < m; i += total)
    {
        int cur_bp = row_ptr[i] / STRIDE;
        int next_bp = row_ptr[i + 1] / STRIDE;

        if (cur_bp != next_bp)
        {
            for (int j = cur_bp + 1; j <= next_bp && j < WGS; j++)
                bp[j] = i;

            if ((row_ptr[i + 1] % STRIDE == 0) && (next_bp < WGS))
                bp[next_bp] = bp[next_bp] + 1;
        }
    }
}

template <int R>
__global__ void spmv_flat(int m,
                          int nnz,
                          const int *__restrict__ row_ptr,
                          const int *__restrict__ col_idx,
                          const float *__restrict__ values,
                          const float *__restrict__ x,
                          float *__restrict__ y,
                          const int *__restrict__ bp,
                          int WGS)
{
    extern __shared__ float shared_mem[];

    const int THREADS = blockDim.x;
    const int N = R * THREADS;
    const int tid = threadIdx.x;
    const int wg_id = blockIdx.x;
    const int wg_nnz_start = wg_id * N;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < R; i++)
    {
        int shared_idx = tid + i * THREADS;
        int global_idx = wg_nnz_start + shared_idx;

        int idx = min(global_idx, nnz - 1);
        float temp = values[idx] * x[col_idx[idx]];
        shared_mem[shared_idx] = (global_idx < nnz) ? temp : 0.0f;
    }

    __syncthreads();

    int bp_idx = wg_id;

    int reduce_row_start = min(bp[bp_idx], m);
    int reduce_row_end = (bp_idx + 1 < WGS) ? min(bp[bp_idx + 1], m) : m;

    if (reduce_row_end == 0)
        reduce_row_end = m;

    if (reduce_row_end < m &&
        (row_ptr[reduce_row_end] % N != 0 || reduce_row_start == reduce_row_end))
        reduce_row_end = min(reduce_row_end + 1, m);

    int reduce_row_id = reduce_row_start + tid;
    int bp_nnz_id = bp_idx * N;

    while (reduce_row_id < reduce_row_end)
    {
        float sum = 0.0f;

        int reduce_id_start = max(0, row_ptr[reduce_row_id] - bp_nnz_id);
        int reduce_id_end = min(N, row_ptr[reduce_row_id + 1] - bp_nnz_id);

        for (int i = reduce_id_start; i < reduce_id_end; i++)
            sum += shared_mem[i];

        atomicAdd(&y[reduce_row_id], sum);
        reduce_row_id += THREADS;
    }
}

// ============================================================
// setup / run_once / teardown
// ============================================================

AccSpMVContext spmv_acc_setup(const CSRDevice &d_csr,
                              const CSRHost &h_csr,
                              const float *d_x,
                              float *d_y)
{
    AccSpMVContext ctx{};
    ctx.nrows = d_csr.nrows;
    ctx.ncols = d_csr.ncols;
    ctx.nnz = d_csr.nnz;
    ctx.d_row_ptr = d_csr.row_ptr;
    ctx.d_col_idx = d_csr.col_idx;
    ctx.d_values = d_csr.values;
    ctx.d_x = d_x;
    ctx.d_y = d_y;
    ctx.d_bp = nullptr;

    const int nrows = ctx.nrows;
    const int nnz = ctx.nnz;
    ctx.avg_nnz = (nrows > 0) ? (float)nnz / nrows : 1.0f;

    int max_row_nnz = 0;
    for (int i = 0; i < nrows; i++)
        max_row_nnz = std::max(max_row_nnz, h_csr.row_ptr[i + 1] - h_csr.row_ptr[i]);
    ctx.imbalance = max_row_nnz / ctx.avg_nnz;

    // ----- Heuristic: LINE vs. FLAT (unchanged from D1's spMVacc.cu) -----
    bool use_line = true;
    if (nnz > 1'000'000 && (ctx.imbalance > 32.0f || ctx.avg_nnz > 8.0f))
        use_line = false;
    if (ctx.avg_nnz > 128.0f)
        use_line = false;

    if (use_line)
    {
        ctx.choice = AccKernel::LINE;
        ctx.line_threads = 512;
        const int R = 2;

        if (ctx.avg_nnz >= 24.0f)
            ctx.line_V = 4;
        else if (ctx.avg_nnz >= 8.0f)
            ctx.line_V = 2;
        else
            ctx.line_V = 1;

        ctx.line_rows_per_wg = ctx.line_threads / ctx.line_V;
        ctx.line_wgs = (nrows + ctx.line_rows_per_wg - 1) / ctx.line_rows_per_wg;
        ctx.line_shmem_size = R * ctx.line_threads * sizeof(float);

        printf("[kernel] acc -> LINE-ENHANCE (nnz=%d, avg_nnz=%.2f, imbalance=%.2f, V=%d)\n",
               nnz, ctx.avg_nnz, ctx.imbalance, ctx.line_V);
    }
    else
    {
        ctx.choice = AccKernel::FLAT;
        ctx.flat_threads = 256;
        const int R = 4;
        const int N = R * ctx.flat_threads;

        ctx.flat_wgs = (nnz + N - 1) / N;
        ctx.flat_shmem_size = N * sizeof(float);

        CUDA_CHECK(cudaMalloc(&ctx.d_bp, (ctx.flat_wgs + 1) * sizeof(int)));
        CUDA_CHECK(cudaMemset(ctx.d_bp, 0, (ctx.flat_wgs + 1) * sizeof(int)));

        int prep_threads = 256;
        int prep_blocks = (nrows + prep_threads - 1) / prep_threads;
        flat_preprocess<<<prep_blocks, prep_threads>>>(nrows, d_csr.row_ptr, ctx.d_bp, N, ctx.flat_wgs);
        CUDA_CHECK(cudaDeviceSynchronize()); // bp[] must be ready before any run_once() call

        printf("[kernel] acc -> FLAT (nnz=%d, avg_nnz=%.2f, imbalance=%.2f)\n",
               nnz, ctx.avg_nnz, ctx.imbalance);
    }

    return ctx;
}

void spmv_acc_run_once(AccSpMVContext &ctx)
{
    if (ctx.choice == AccKernel::LINE)
    {
        if (ctx.line_V == 1)
            spmv_line_enhance<2, 1><<<ctx.line_wgs, ctx.line_threads, ctx.line_shmem_size>>>(
                ctx.nrows, ctx.line_rows_per_wg, ctx.d_row_ptr, ctx.d_col_idx, ctx.d_values, ctx.d_x, ctx.d_y);
        else if (ctx.line_V == 2)
            spmv_line_enhance<2, 2><<<ctx.line_wgs, ctx.line_threads, ctx.line_shmem_size>>>(
                ctx.nrows, ctx.line_rows_per_wg, ctx.d_row_ptr, ctx.d_col_idx, ctx.d_values, ctx.d_x, ctx.d_y);
        else
            spmv_line_enhance<2, 4><<<ctx.line_wgs, ctx.line_threads, ctx.line_shmem_size>>>(
                ctx.nrows, ctx.line_rows_per_wg, ctx.d_row_ptr, ctx.d_col_idx, ctx.d_values, ctx.d_x, ctx.d_y);
    }
    else
    {
        spmv_flat<4><<<ctx.flat_wgs, ctx.flat_threads, ctx.flat_shmem_size>>>(
            ctx.nrows, ctx.nnz, ctx.d_row_ptr, ctx.d_col_idx, ctx.d_values, ctx.d_x, ctx.d_y, ctx.d_bp, ctx.flat_wgs);
    }
}

void spmv_acc_teardown(AccSpMVContext &ctx)
{
    if (ctx.d_bp)
        CUDA_CHECK(cudaFree(ctx.d_bp));
}

const char *spmv_acc_kernel_name(const AccSpMVContext &ctx)
{
    return (ctx.choice == AccKernel::LINE) ? "LINE-ENHANCE" : "FLAT";
}

long long spmv_acc_bytes_moved(const AccSpMVContext &ctx)
{
    long long base = (long long)ctx.nnz * sizeof(float)          // values[]
                    + (long long)ctx.nnz * sizeof(int)            // col_idx[]
                    + (long long)(ctx.nrows + 1) * sizeof(int)     // row_ptr[]
                    + (long long)ctx.ncols * sizeof(float)          // x[]
                    + (long long)ctx.nrows * sizeof(float);          // y[]

    if (ctx.choice == AccKernel::FLAT)
        base += (long long)ctx.flat_wgs * sizeof(int); // bp[] read per call

    return base;
}
