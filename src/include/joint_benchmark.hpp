#pragma once
#include <algorithm>
#include <cstdio>
#include <mpi.h>
#include <string>
#include <utility>
#include <vector>

// Shared warmup+benchmark harness: measures a per-SpMV-call pipeline
// (vector exchange + local kernel) jointly, as a single timed unit.
const int JOINT_WARMUP_ITERATIONS = 10;
const int JOINT_BENCHMARK_ITERATIONS = 30;

struct BenchStats
{
    double avg_ms = 0.0;
    double min_ms = 0.0;
    double max_ms = 0.0;
    double variance_ms = 0.0;
};

inline BenchStats summarize(const std::vector<double> &samples)
{
    BenchStats r;
    if (samples.empty())
        return r;

    double total = 0.0, total2 = 0.0;
    r.min_ms = samples[0];
    r.max_ms = samples[0];
    for (double v : samples)
    {
        total += v;
        total2 += v * v;
        r.min_ms = std::min(r.min_ms, v);
        r.max_ms = std::max(r.max_ms, v);
    }
    r.avg_ms = total / samples.size();
    r.variance_ms = total2 / samples.size() - r.avg_ms * r.avg_ms;
    return r;
}

// iteration_fn: performs exactly one exchange+kernel cycle, timing both
// itself, and returns {comm_ms, compute_ms} for that cycle. This wrapper
// supplies the warmup/benchmark looping and per-iteration barriers (so
// every rank enters each rep together).
//
// Reports comm-only and compute-only stats, plus TOTAL stats computed from
// the per-rep SUM (comm_ms_i + compute_ms_i) directly -- total_stats's
// min/max are not derivable from the separately-averaged comm/compute
// stats, since the worst comm rep and worst compute rep need not coincide.
template <typename IterFn>
void run_joint_benchmark(MPI_Comm comm, IterFn &&iteration_fn,
                          BenchStats &comm_stats, BenchStats &compute_stats, BenchStats &total_stats)
{
    for (int i = 0; i < JOINT_WARMUP_ITERATIONS; i++)
    {
        MPI_Barrier(comm);
        iteration_fn();
    }

    std::vector<double> comm_samples, compute_samples, total_samples;
    comm_samples.reserve(JOINT_BENCHMARK_ITERATIONS);
    compute_samples.reserve(JOINT_BENCHMARK_ITERATIONS);
    total_samples.reserve(JOINT_BENCHMARK_ITERATIONS);

    for (int i = 0; i < JOINT_BENCHMARK_ITERATIONS; i++)
    {
        MPI_Barrier(comm);
        std::pair<double, double> sample = iteration_fn();
        comm_samples.push_back(sample.first);
        compute_samples.push_back(sample.second);
        total_samples.push_back(sample.first + sample.second);
    }

    comm_stats = summarize(comm_samples);
    compute_stats = summarize(compute_samples);
    total_stats = summarize(total_samples);
}

inline std::string basename_of(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

// Self-describing per-rank result line, shared by every GPU/MPI driver.
// Analysis scripts parse this as "key=value" pairs, not positional tokens.
//
//   comm       communication strategy: bcast | ghost_mpi | ghost_nccl |
//              checkerboard_mpi | checkerboard_nccl | rcm_nccl | fennel_nccl |
//              block_nccl
//   kernel     local SpMV kernel: cusparse | acc
//   n_ghost    -1 for drivers with no ghost concept (bcast, checkerboard)
inline void print_result(const char *comm, const char *kernel, const std::string &matrix,
                          int P, int rank,
                          int nrows_local, int ncols_global, int nnz_local, int n_ghost,
                          const BenchStats &compute_stats, double compute_gflops,
                          double eff_bw_gbs, double pct_peak_bw,
                          const BenchStats &comm_stats,
                          const BenchStats &total_stats, double effective_gflops)
{
    printf("RESULT comm=%s kernel=%s matrix=%s P=%d rank=%d "
           "nrows_local=%d ncols_global=%d nnz_local=%d n_ghost=%d "
           "compute_avg_ms=%.6f compute_min_ms=%.6f compute_max_ms=%.6f compute_var_ms=%.9f "
           "compute_gflops=%.6f eff_bw_gbs=%.6f pct_peak_bw=%.4f "
           "comm_avg_ms=%.6f comm_min_ms=%.6f comm_max_ms=%.6f comm_var_ms=%.9f "
           "total_avg_ms=%.6f total_min_ms=%.6f total_max_ms=%.6f total_var_ms=%.9f "
           "effective_gflops=%.6f\n",
           comm, kernel, basename_of(matrix).c_str(), P, rank,
           nrows_local, ncols_global, nnz_local, n_ghost,
           compute_stats.avg_ms, compute_stats.min_ms, compute_stats.max_ms, compute_stats.variance_ms,
           compute_gflops, eff_bw_gbs, pct_peak_bw,
           comm_stats.avg_ms, comm_stats.min_ms, comm_stats.max_ms, comm_stats.variance_ms,
           total_stats.avg_ms, total_stats.min_ms, total_stats.max_ms, total_stats.variance_ms,
           effective_gflops);
    fflush(stdout);
}
