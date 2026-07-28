/**
 * Identical communication code to spmv_bcast.cu (full MPI_Bcast of x every
 * call, no selective exchange), ACC kernel (LINE/FLAT, chosen per-matrix at
 * setup time) instead of cuSPARSE.
 *
 * Usage:
 *   mpirun -n <P> ./bin/spmv_bcast_acc <matrix.mtx>
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>

#include "matrix.hpp"
#include "cpu.hpp"
#include "random.hpp"
#include "cuda_check.hpp"
#include "gpu_binding.hpp"
#include "acc.hpp"
#include "distribute_matrix.hpp"
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

    // ---- Rank 0: read matrix, build reference CSR, generate x ----
    int nrows = 0, ncols = 0;
    std::vector<COO> coo;
    CSRHost csr_ref;
    std::vector<float> x_full;

    if (rank == 0)
    {
        bool symmetric;
        coo = read_mtx(path, nrows, ncols, symmetric);

        std::vector<COO> coo_copy = coo; // coo_to_csr sorts in place
        csr_ref = coo_to_csr(coo_copy, nrows, ncols);

        x_full = generateRandomArray(ncols);

        printf("Loaded %s: %d x %d, %d nnz\n", path.c_str(), nrows, ncols, csr_ref.nnz);
    }

    LocalCSR local = distribute_matrix(coo, nrows, ncols, 0, MPI_COMM_WORLD);

    if (rank != 0)
        x_full.resize(local.ncols_global);

    CSRHost h;
    h.nrows = local.nrows_local;
    h.ncols = local.ncols_global;
    h.nnz   = local.nnz_local;
    h.row_ptr = std::move(local.row_ptr);
    h.col_idx = std::move(local.col_idx);
    h.values  = std::move(local.values);
    CSRDevice d_csr = csr_host_to_device(h); // h itself is untouched (const ref), reused below

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, (size_t)local.ncols_global * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)local.nrows_local * sizeof(float)));

    // Bound to d_x/d_y once; d_x's CONTENTS are refreshed every iteration
    // by the Bcast below, the pointer itself never changes.
    AccSpMVContext ctx = spmv_acc_setup(d_csr, h, d_x, d_y);

    cudaEvent_t k_start, k_stop;
    CUDA_CHECK(cudaEventCreate(&k_start));
    CUDA_CHECK(cudaEventCreate(&k_stop));

    auto iteration = [&]() -> std::pair<double, double>
    {
        double t0 = MPI_Wtime();
        MPI_Bcast(x_full.data(), local.ncols_global, MPI_FLOAT, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_x, x_full.data(), (size_t)local.ncols_global * sizeof(float),
                              cudaMemcpyHostToDevice));
        double comm_ms = (MPI_Wtime() - t0) * 1000.0;

        CUDA_CHECK(cudaMemset(d_y, 0, (size_t)local.nrows_local * sizeof(float)));
        CUDA_CHECK(cudaEventRecord(k_start));
        spmv_acc_run_once(ctx);
        CUDA_CHECK(cudaEventRecord(k_stop));
        CUDA_CHECK(cudaEventSynchronize(k_stop));
        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k_start, k_stop));

        return {comm_ms, (double)kernel_ms};
    };

    BenchStats comm_stats, compute_stats, total_stats;
    run_joint_benchmark(MPI_COMM_WORLD, iteration, comm_stats, compute_stats, total_stats);

    spmv_acc_teardown(ctx);
    CUDA_CHECK(cudaEventDestroy(k_start));
    CUDA_CHECK(cudaEventDestroy(k_stop));

    std::vector<float> y_local(local.nrows_local);
    CUDA_CHECK(cudaMemcpy(y_local.data(), d_y, (size_t)local.nrows_local * sizeof(float),
                          cudaMemcpyDeviceToHost));

    long long bytes_moved = spmv_acc_bytes_moved(ctx);
    double compute_gflops = 2.0 * local.nnz_local / (compute_stats.avg_ms / 1e3) / 1e9;
    double effective_gflops = 2.0 * local.nnz_local / (total_stats.avg_ms / 1e3) / 1e9;
    double eff_bw_gbs = (compute_stats.avg_ms > 0.0)
                             ? (double)bytes_moved / (compute_stats.avg_ms * 1e-3) / 1e9
                             : 0.0;

    for (int rr = 0; rr < size; rr++)
    {
        if (rr == rank)
            print_result("bcast", "acc", path, size, rank,
                         local.nrows_local, local.ncols_global, local.nnz_local, -1,
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
            recvcounts[rr] = local_nrows(local.nrows_global, size, rr);
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
                int global_row = lr * size + rr;
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
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));

    MPI_Finalize();
    return 0;
}
