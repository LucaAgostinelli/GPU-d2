/**
 * NCCL 2D checkerboard (identical to spmv_2d_nccl_cusparse.cu) with the ACC
 * kernel instead of cuSPARSE
 *
 * Usage:
 *   mpirun -n <P> ./bin/spmv_2d_nccl_acc <matrix.mtx>
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "matrix.hpp"
#include "cpu.hpp"
#include "random.hpp"
#include "cuda_check.hpp"
#include "gpu_binding.hpp"
#include "acc.hpp"
#include "distribute_matrix.hpp" // local_nrows()
#include "distribute_matrix_2d.hpp"
#include "proc_grid.hpp"
#include "nccl_bootstrap.hpp"
#include "joint_benchmark.hpp"

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

    ProcGrid grid = make_proc_grid(MPI_COMM_WORLD);
    if (rank == 0)
        printf("Grid: P=%d -> P_r=%d x P_c=%d%s\n", grid.P, grid.P_r, grid.P_c,
               (grid.P_r == 1 && grid.P > 1) ? " (degenerate 1D grid)" : "");

    ncclComm_t nccl_col_comm = bootstrap_nccl_comm(grid.col_comm);
    ncclComm_t nccl_row_comm = bootstrap_nccl_comm(grid.row_comm);

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

    LocalCSR2D local = distribute_matrix_2d(coo, nrows, ncols, grid, 0, MPI_COMM_WORLD);

    std::vector<int> x_sendcounts(size, 0), x_displs(size, 0);
    std::vector<float> x_flat;
    if (rank == 0)
    {
        int off = 0;
        for (int c = 0; c < grid.P_c; c++)
        {
            int ncols_c = local_nrows(local.ncols_global, grid.P_c, c);
            x_sendcounts[c] = ncols_c;
            x_displs[c] = off;
            off += ncols_c;
        }
        x_flat.resize(off);
        for (int c = 0; c < grid.P_c; c++)
        {
            int ncols_c = x_sendcounts[c];
            for (int lc = 0; lc < ncols_c; lc++)
            {
                int global_j = lc * grid.P_c + c;
                x_flat[x_displs[c] + lc] = x_full[global_j];
            }
        }
    }

    int x_recv_n = (grid.pr == 0) ? local.ncols_local : 0;
    std::vector<float> x_seg_init(local.ncols_local, 0.0f);
    MPI_Scatterv(rank == 0 ? x_flat.data() : nullptr,
                 rank == 0 ? x_sendcounts.data() : nullptr,
                 rank == 0 ? x_displs.data() : nullptr,
                 MPI_FLOAT,
                 x_seg_init.data(), x_recv_n, MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    float *d_x_full = nullptr, *d_y_partial = nullptr, *d_y_final = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x_full, (size_t)local.ncols_local * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x_full, x_seg_init.data(),
                          (size_t)local.ncols_local * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_y_partial, (size_t)local.nrows_local * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_final,   (size_t)local.nrows_local * sizeof(float)));

    auto do_bcast = [&]()
    {
        NCCL_CHECK(ncclBroadcast(d_x_full, d_x_full, local.ncols_local, ncclFloat,
                                  0, nccl_col_comm, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
    };

    auto do_reduce = [&]()
    {
        NCCL_CHECK(ncclReduce(d_y_partial, d_y_final, local.nrows_local, ncclFloat,
                               ncclSum, 0, nccl_row_comm, 0));
        CUDA_CHECK(cudaStreamSynchronize(0));
    };

    CSRHost h;
    h.nrows = local.nrows_local;
    h.ncols = local.ncols_local;
    h.nnz   = local.nnz_local;
    h.row_ptr = std::move(local.row_ptr);
    h.col_idx = std::move(local.col_idx);
    h.values  = std::move(local.values);
    CSRDevice d_csr = csr_host_to_device(h);

    AccSpMVContext ctx = spmv_acc_setup(d_csr, h, d_x_full, d_y_partial);

    cudaEvent_t k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));

    auto iteration = [&]() -> std::pair<double, double>
    {
        double t0 = MPI_Wtime();
        do_bcast();
        double bcast_ms = (MPI_Wtime() - t0) * 1000.0;

        CUDA_CHECK(cudaMemset(d_y_partial, 0, (size_t)local.nrows_local * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(k_start));
        spmv_acc_run_once(ctx);
        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k_start, k_stop));

        double t1 = MPI_Wtime();
        do_reduce();
        double reduce_ms = (MPI_Wtime() - t1) * 1000.0;

        return {bcast_ms + reduce_ms, (double)kernel_ms};
    };

    BenchStats comm_stats, compute_stats, total_stats;
    run_joint_benchmark(MPI_COMM_WORLD, iteration, comm_stats, compute_stats, total_stats);

    spmv_acc_teardown(ctx);
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));

    long long bytes_moved = spmv_acc_bytes_moved(ctx);
    double compute_gflops = 2.0 * local.nnz_local / (compute_stats.avg_ms / 1e3) / 1e9;
    double effective_gflops = 2.0 * local.nnz_local / (total_stats.avg_ms / 1e3) / 1e9;
    double eff_bw_gbs = (compute_stats.avg_ms > 0.0)
                             ? (double)bytes_moved / (compute_stats.avg_ms * 1e-3) / 1e9
                             : 0.0;

    for (int rr = 0; rr < size; rr++)
    {
        if (rr == rank)
            print_result("checkerboard_nccl", "acc", path, size, rank,
                         local.nrows_local, local.ncols_global, local.nnz_local, -1,
                         compute_stats, compute_gflops,
                         eff_bw_gbs, 100.0 * eff_bw_gbs / peak_bw_gbs,
                         comm_stats, total_stats, effective_gflops);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int y_send_n = (grid.pc == 0) ? local.nrows_local : 0;
    std::vector<float> y_row(std::max(y_send_n, 1));
    if (y_send_n > 0)
        CUDA_CHECK(cudaMemcpy(y_row.data(), d_y_final,
                              (size_t)y_send_n * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<int> recvcounts(size), displs(size);
    std::vector<float> y_flat;
    if (rank == 0)
    {
        int off = 0;
        for (int r = 0; r < size; r++)
        {
            int pr_r = r / grid.P_c;
            int pc_r = r % grid.P_c;
            recvcounts[r] = (pc_r == 0) ? local_nrows(local.nrows_global, grid.P_r, pr_r) : 0;
            displs[r] = off;
            off += recvcounts[r];
        }
        y_flat.resize(off);
    }

    MPI_Gatherv(y_row.data(), y_send_n, MPI_FLOAT,
                rank == 0 ? y_flat.data() : nullptr,
                rank == 0 ? recvcounts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        std::vector<float> y_global(local.nrows_global);
        for (int r = 0; r < size; r++)
        {
            if (recvcounts[r] == 0)
                continue;
            int pr_r = r / grid.P_c;
            int nrows_r = recvcounts[r];
            for (int lr = 0; lr < nrows_r; lr++)
            {
                int global_row = lr * grid.P_r + pr_r;
                y_global[global_row] = y_flat[displs[r] + lr];
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
    CUDA_CHECK(cudaFree(d_y_partial));
    CUDA_CHECK(cudaFree(d_y_final));

    NCCL_CHECK(ncclCommDestroy(nccl_col_comm));
    NCCL_CHECK(ncclCommDestroy(nccl_row_comm));

    free_proc_grid(grid);
    MPI_Finalize();
    return 0;
}
