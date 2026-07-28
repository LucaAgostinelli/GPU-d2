/**
 * Read-phase benchmark comparing two strategies for loading a .mtx file
 * into a distributed LocalCSR shard on every rank:
 *
 *   - "serial":  rank 0 reads the whole file (read_mtx, common/matrix.cpp)
 *     and scatters pre-built CSR shards via Scatterv (distribute_matrix.cpp).
 *   - "chunked": every rank independently opens the file, parses the
 *     header, and reads its own line-aligned byte range with ifstream::read
 *     (distribute_matrix_chunked.cpp), then redistributes by
 *     owner(i) = i % P via MPI_Alltoallv, matching the ownership rule used
 *     by "serial".
 *
 * This benchmark covers only the read+distribute phase, which is independent of the downstream SpMV
 * kernel or communication strategy, and is not wired into any SpMV driver.
 *
 * Usage:
 *   mpirun -n <P> ./bin/bench_mtx_read <matrix.mtx> [reps]
 *
 * Rank 0 prints one RESULT_CSV_READ line containing mean/min timings for
 * both methods, a phase breakdown for "chunked" (io/parse/redistribute),
 * and a numeric correctness check: a fixed deterministic x vector is
 * multiplied against each method's LocalCSR shard on every rank, and the
 * resulting y vectors are compared element-by-element using the same
 * tolerance (almost_equal, common/cpu.cpp) used by the project's CPU
 * reference checks.
 */
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>

#include <mpi.h>

#include "matrix.hpp"
#include "distribute_matrix.hpp"
#include "distribute_matrix_chunked.hpp"
#include "cpu.hpp"
#include "random.hpp"

// SpMV against a LocalCSR shard directly (global column indices index
// straight into the full x vector). Mirrors spmv_cpu (common/cpu.cpp),
// but avoids copying a LocalCSR's row_ptr/col_idx/values into a separate
// CSRHost first.
static void spmv_local(const LocalCSR &csr, const std::vector<float> &x, std::vector<float> &y)
{
    y.resize(csr.nrows_local);
    for (int row = 0; row < csr.nrows_local; row++)
    {
        float sum = 0.0f;
        for (int k = csr.row_ptr[row]; k < csr.row_ptr[row + 1]; k++)
            sum += csr.values[k] * x[csr.col_idx[k]];
        y[row] = sum;
    }
}

struct SerialMetrics
{
    double read_ms = 0.0;       // read_mtx (rank 0 only; 0 on every other rank)
    double distribute_ms = 0.0; // distribute_matrix (bucket+CSR on root, Scatterv on all)
};

static double time_serial(const std::string &path, MPI_Comm comm, LocalCSR &out, SerialMetrics &metrics)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    std::vector<COO> coo;
    int nrows = 0, ncols = 0;
    bool symmetric = false;

    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    if (rank == 0)
        coo = read_mtx(path, nrows, ncols, symmetric);
    double t_mid = MPI_Wtime();

    out = distribute_matrix(coo, nrows, ncols, 0, comm);
    double t1 = MPI_Wtime();

    MPI_Barrier(comm);
    double t_end = MPI_Wtime();

    metrics.read_ms = (t_mid - t0) * 1000.0;
    metrics.distribute_ms = (t1 - t_mid) * 1000.0;
    return (t_end - t0) * 1000.0;
}

static double time_chunked(const std::string &path, MPI_Comm comm, LocalCSR &out, ChunkedReadMetrics &metrics)
{
    MPI_Barrier(comm);
    double t0 = MPI_Wtime();

    out = read_and_distribute_chunked(path, comm, &metrics);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    return (t1 - t0) * 1000.0;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

    if (argc < 2)
    {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <matrix.mtx> [reps]\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    std::string path = argv[1];
    int reps = (argc >= 3) ? std::atoi(argv[2]) : 3;

    std::string matrix_name = path.substr(path.find_last_of("/\\") + 1);

    std::vector<double> serial_ms, chunked_ms;
    std::vector<double> serial_read_ms, serial_distribute_ms;
    std::vector<double> chunked_io_ms, chunked_parse_ms, chunked_redist_ms;
    LocalCSR serial_out{}, chunked_out{};

    // Untimed warmup run per method, executed before any timed repetition.
    {
        LocalCSR warm_out;
        SerialMetrics warm_metrics;
        time_serial(path, MPI_COMM_WORLD, warm_out, warm_metrics);
    }
    {
        LocalCSR warm_out;
        ChunkedReadMetrics warm_metrics;
        time_chunked(path, MPI_COMM_WORLD, warm_out, warm_metrics);
    }

    for (int r = 0; r < reps; r++)
    {
        SerialMetrics m;
        serial_ms.push_back(time_serial(path, MPI_COMM_WORLD, serial_out, m));
        serial_read_ms.push_back(m.read_ms);
        serial_distribute_ms.push_back(m.distribute_ms);
    }

    for (int r = 0; r < reps; r++)
    {
        ChunkedReadMetrics m;
        chunked_ms.push_back(time_chunked(path, MPI_COMM_WORLD, chunked_out, m));
        chunked_io_ms.push_back(m.io_ms);
        chunked_parse_ms.push_back(m.parse_ms);
        chunked_redist_ms.push_back(m.redistribute_ms);
    }

    auto mean = [](const std::vector<double> &v)
    { return std::accumulate(v.begin(), v.end(), 0.0) / v.size(); };
    auto min_of = [](const std::vector<double> &v)
    { return *std::min_element(v.begin(), v.end()); };

    // Each rank computes its own phase-breakdown means; MPI_Reduce with MAX
    // picks the slowest rank per phase, since that determines wall-clock
    // time under barrier synchronization. serial's read_ms is 0 on every
    // non-root rank, so MAX naturally recovers rank 0's value there.
    double local_read_mean = mean(serial_read_ms);
    double local_distribute_mean = mean(serial_distribute_ms);
    double local_io_mean = mean(chunked_io_ms);
    double local_parse_mean = mean(chunked_parse_ms);
    double local_redist_mean = mean(chunked_redist_ms);

    double read_mean = 0, distribute_mean = 0, io_mean = 0, parse_mean = 0, redist_mean = 0;
    MPI_Reduce(&local_read_mean, &read_mean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_distribute_mean, &distribute_mean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_io_mean, &io_mean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_parse_mean, &parse_mean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_redist_mean, &redist_mean, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Cross-check, step 1: verify both methods agree on global nnz and on
    // this rank's own shard shape. Results are reduced to every rank via
    // Allreduce, not just rank 0, and checked before the numeric SpMV
    // comparison below, which indexes row_ptr/col_idx assuming matching
    // shapes and would read out of bounds otherwise.
    long long nnz_local_serial = serial_out.nnz_local;
    long long nnz_local_chunked = chunked_out.nnz_local;
    long long nnz_total_serial = 0, nnz_total_chunked = 0;
    MPI_Reduce(&nnz_local_serial, &nnz_total_serial, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&nnz_local_chunked, &nnz_total_chunked, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    bool shard_match = (serial_out.nrows_global == chunked_out.nrows_global) &&
                       (serial_out.ncols_global == chunked_out.ncols_global) &&
                       (serial_out.nrows_local == chunked_out.nrows_local) &&
                       (serial_out.nnz_local == chunked_out.nnz_local);
    int local_ok = shard_match ? 1 : 0, shapes_ok = 0;
    MPI_Allreduce(&local_ok, &shapes_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Cross-check, step 2: numeric SpMV comparison
    long long total_mismatches = -1; // sentinel: shape mismatch, never computed
    if (shapes_ok)
    {
        std::vector<float> x = generateRandomArray(serial_out.ncols_global);
        std::vector<float> y_serial, y_chunked;
        spmv_local(serial_out, x, y_serial);
        spmv_local(chunked_out, x, y_chunked);

        long long local_mismatches = 0;
        for (int i = 0; i < serial_out.nrows_local; i++)
            if (!almost_equal(y_serial[i], y_chunked[i]))
                local_mismatches++;

        long long global_mismatches = 0;
        MPI_Reduce(&local_mismatches, &global_mismatches, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        total_mismatches = global_mismatches;
    }

    if (rank == 0)
    {
        bool nnz_match = (nnz_total_serial == nnz_total_chunked);
        bool ok = nnz_match && (shapes_ok == 1) && (total_mismatches == 0);

        printf("RESULT_CSV_READ matrix=%s P=%d reps=%d "
               "serial_mean_ms=%.4f serial_min_ms=%.4f "
               "chunked_mean_ms=%.4f chunked_min_ms=%.4f "
               "serial_read_ms=%.4f serial_distribute_ms=%.4f "
               "chunked_io_ms=%.4f chunked_parse_ms=%.4f chunked_redistribute_ms=%.4f "
               "nnz_total_serial=%lld nnz_total_chunked=%lld "
               "spmv_mismatches=%lld "
               "check=%s\n",
               matrix_name.c_str(), P, reps,
               mean(serial_ms), min_of(serial_ms),
               mean(chunked_ms), min_of(chunked_ms),
               read_mean, distribute_mean,
               io_mean, parse_mean, redist_mean,
               nnz_total_serial, nnz_total_chunked,
               total_mismatches,
               ok ? "OK!" : "MISMATCH!");
    }

    MPI_Finalize();
    return 0;
}