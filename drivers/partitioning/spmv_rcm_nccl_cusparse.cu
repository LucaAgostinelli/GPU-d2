/**
 * Twin of spmv_rcm_nccl_acc.cu -- IDENTICAL RCM partitioning
 * and NCCL ghost-exchange transport, cuSPARSE instead of ACC.
 *
 * Usage:
 *   mpirun -n <P> ./bin/spmv_rcm_nccl_cusparse <matrix.mtx>
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "matrix.hpp"
#include "cpu.hpp"
#include "random.hpp"
#include "cuda_check.hpp"
#include "gpu_binding.hpp"
#include "cusparse.hpp"
#include "distribute_matrix.hpp"
#include "rcm_partition.hpp"
#include "distribute_matrix_partitioned.hpp"
#include "ghost_exchange_mapped.hpp"
#include "nccl_bootstrap.hpp"
#include "joint_benchmark.hpp"

__global__ void gather_kernel(const float *src, const int *idx, float *dst, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        dst[i] = src[idx[i]];
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <matrix.mtx>\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::string path = argv[1];

    float peak_bw_gbs = bind_gpu_and_get_peak_bw(rank);

    ncclComm_t nccl_comm = bootstrap_nccl_comm(MPI_COMM_WORLD);

    int nrows = 0, ncols = 0;
    std::vector<COO> coo;
    CSRHost csr_ref;
    std::vector<float> x_full;

    if (rank == 0)
    {
        bool symmetric;
        coo = read_mtx(path, nrows, ncols, symmetric);

        std::vector<COO> coo_copy = coo;
        csr_ref = coo_to_csr(coo_copy, nrows, ncols);

        x_full = generateRandomArray(ncols);

        printf("Loaded %s: %d x %d, %d nnz\n", path.c_str(), nrows, ncols, csr_ref.nnz);
    }

    RcmPartition rp = compute_rcm_partition(coo, nrows, 0, MPI_COMM_WORLD);

    LocalCSR local = distribute_matrix_partitioned(coo, nrows, ncols,
                                                    rp.owner_of, rp.local_of, rp.block_start,
                                                    0, MPI_COMM_WORLD);
    GhostPlan plan = build_ghost_plan_mapped(local, rp.owner_of, rp.local_of, MPI_COMM_WORLD);

    std::vector<int> x_sendcounts(size), x_displs(size);
    std::vector<float> x_flat;
    if (rank == 0)
    {
        int off = 0;
        for (int r = 0; r < size; r++)
        {
            x_sendcounts[r] = rp.block_start[r + 1] - rp.block_start[r];
            x_displs[r] = off;
            off += x_sendcounts[r];
        }
        x_flat.resize(off);
        for (int r = 0; r < size; r++)
        {
            int nrows_r = x_sendcounts[r];
            for (int lr = 0; lr < nrows_r; lr++)
            {
                int global_i = rp.order[rp.block_start[r] + lr];
                x_flat[x_displs[r] + lr] = x_full[global_i];
            }
        }
    }

    std::vector<float> x_owned(local.nrows_local);
    MPI_Scatterv(rank == 0 ? x_flat.data() : nullptr,
                 rank == 0 ? x_sendcounts.data() : nullptr,
                 rank == 0 ? x_displs.data() : nullptr,
                 MPI_FLOAT,
                 x_owned.data(), local.nrows_local, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    int ncols_local = local.nrows_local + plan.n_ghost;
    float *d_x_full = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_full, (size_t)ncols_local * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_full, x_owned.data(),
                          (size_t)local.nrows_local * sizeof(float), cudaMemcpyHostToDevice));

    int n_send = (int)plan.requested_of_me.size();
    int *d_send_idx = nullptr;
    float *d_sendbuf = nullptr;
    if (n_send > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_send_idx, (size_t)n_send * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_send_idx, plan.requested_of_me.data(),
                              (size_t)n_send * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_sendbuf, (size_t)n_send * sizeof(float)));
    }

    auto do_exchange = [&]()
    {
        if (n_send > 0)
        {
            int threads = 256;
            int blocks = (n_send + threads - 1) / threads;
            gather_kernel<<<blocks, threads>>>(d_x_full, d_send_idx, d_sendbuf, n_send);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        NCCL_CHECK(ncclGroupStart());
        for (int p = 0; p < size; p++)
        {
            if (plan.need_counts[p] > 0)
                NCCL_CHECK(ncclRecv(d_x_full + local.nrows_local + plan.need_displs[p],
                                     plan.need_counts[p], ncclFloat, p, nccl_comm, 0));
        }
        for (int p = 0; p < size; p++)
        {
            if (plan.req_counts[p] > 0)
                NCCL_CHECK(ncclSend(d_sendbuf + plan.req_displs[p],
                                     plan.req_counts[p], ncclFloat, p, nccl_comm, 0));
        }
        NCCL_CHECK(ncclGroupEnd());
        CUDA_CHECK(cudaStreamSynchronize(0));
    };

    CSRHost h;
    h.nrows = local.nrows_local;
    h.ncols = ncols_local;
    h.nnz   = local.nnz_local;
    h.row_ptr = std::move(local.row_ptr);
    h.col_idx = std::move(local.col_idx);
    h.values  = std::move(local.values);
    CSRDevice d_csr = csr_host_to_device(h);

    CUDA_CHECK(cudaMalloc(&d_y, (size_t)local.nrows_local * sizeof(float)));

    CusparseSpMVContext ctx = spmv_cusparse_setup(d_csr, d_x_full, d_y);

    cudaEvent_t k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));

    auto iteration = [&]() -> std::pair<double, double>
    {
        double t0 = MPI_Wtime();
        do_exchange();
        double comm_ms = (MPI_Wtime() - t0) * 1000.0;

        CUDA_CHECK(cudaMemset(d_y, 0, (size_t)local.nrows_local * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(k_start));
        spmv_cusparse_run_once(ctx);
        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k_start, k_stop));

        return {comm_ms, (double)kernel_ms};
    };

    BenchStats comm_stats, compute_stats, total_stats;
    run_joint_benchmark(MPI_COMM_WORLD, iteration, comm_stats, compute_stats, total_stats);

    spmv_cusparse_teardown(ctx);
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));

    std::vector<float> y_local(local.nrows_local);
    CUDA_CHECK(cudaMemcpy(y_local.data(), d_y, (size_t)local.nrows_local * sizeof(float),
                          cudaMemcpyDeviceToHost));

    long long bytes_moved = spmv_cusparse_bytes_moved(d_csr);
    double compute_gflops = 2.0 * local.nnz_local / (compute_stats.avg_ms / 1e3) / 1e9;
    double effective_gflops = 2.0 * local.nnz_local / (total_stats.avg_ms / 1e3) / 1e9;
    double eff_bw_gbs = (compute_stats.avg_ms > 0.0)
                             ? (double)bytes_moved / (compute_stats.avg_ms * 1e-3) / 1e9
                             : 0.0;

    for (int rr = 0; rr < size; rr++)
    {
        if (rr == rank)
            print_result("rcm_nccl", "cusparse", path, size, rank,
                         local.nrows_local, local.ncols_global, local.nnz_local, plan.n_ghost,
                         compute_stats, compute_gflops,
                         eff_bw_gbs, 100.0 * eff_bw_gbs / peak_bw_gbs,
                         comm_stats, total_stats, effective_gflops);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    std::vector<int> recvcounts(size), displs(size);
    std::vector<float> y_flat;
    if (rank == 0)
    {
        int off = 0;
        for (int rr = 0; rr < size; rr++)
        {
            recvcounts[rr] = rp.block_start[rr + 1] - rp.block_start[rr];
            displs[rr] = off;
            off += recvcounts[rr];
        }
        y_flat.resize(off);
    }

    MPI_Gatherv(y_local.data(), local.nrows_local, MPI_FLOAT,
                rank == 0 ? y_flat.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::vector<float> y_global(local.nrows_global);
        for (int rr = 0; rr < size; rr++)
        {
            int nrows_r = recvcounts[rr];
            for (int lr = 0; lr < nrows_r; lr++)
            {
                int global_row = rp.order[rp.block_start[rr] + lr];
                y_global[global_row] = y_flat[displs[rr] + lr];
            }
        }

        std::vector<float> y_cpu(nrows, 0.0f);
        spmv_cpu(nrows, csr_ref, x_full, y_cpu);
        check_correctness(nrows, y_cpu, y_global);
    }

    CUDA_CHECK(cudaFree(d_csr.row_ptr));
    CUDA_CHECK(cudaFree(d_csr.col_idx));
    CUDA_CHECK(cudaFree(d_csr.values));
    CUDA_CHECK(cudaFree(d_x_full));
    CUDA_CHECK(cudaFree(d_y));
    if (d_send_idx) CUDA_CHECK(cudaFree(d_send_idx));
    if (d_sendbuf)  CUDA_CHECK(cudaFree(d_sendbuf));

    NCCL_CHECK(ncclCommDestroy(nccl_comm));

    MPI_Finalize();
    return 0;
}
