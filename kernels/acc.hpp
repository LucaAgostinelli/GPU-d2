#pragma once
#include <cuda_runtime.h>
#include "matrix.hpp"

// Single-shot version of D1's ACC dispatcher (spMVacc.cu): picks LINE vs.
// FLAT once at setup time from the matrix's nnz / avg-nnz-per-row / row
// imbalance, then exposes a single run_once() kernel launch. The underlying kernel code
// (spmv_line_enhance<R,V>, spmv_flat<R>, and the LINE/FLAT parameter
// heuristics) is unchanged from D1's spMVline.cu/spMVflat.cu/spMVacc.cu
enum class AccKernel
{
    LINE,
    FLAT
};

struct AccSpMVContext
{
    AccKernel choice;

    int nrows, ncols, nnz;
    const int *d_row_ptr;
    const int *d_col_idx;
    const float *d_values;
    const float *d_x;
    float *d_y;

    // LINE launch parameters
    int line_threads;
    int line_V;
    int line_rows_per_wg;
    int line_wgs;
    int line_shmem_size;

    // FLAT launch parameters
    int flat_threads;
    int flat_wgs;
    int flat_shmem_size;
    int *d_bp; // breakpoint array, built once here at setup

    // Diagnostics, for the one setup-time print and for reporting alongside
    // the result line.
    float avg_nnz;
    float imbalance;
};

// One-time setup: runs the LINE-vs-FLAT heuristic (needs h_csr's row_ptr
// for the imbalance calculation -- d_csr alone doesn't expose row lengths
// without a device->host round trip), computes launch parameters for
// whichever kernel is chosen, and -- for FLAT specifically -- builds the
// bp[] breakpoint array once here via flat_preprocess. Prints one line
// naming the chosen kernel and why.
AccSpMVContext spmv_acc_setup(const CSRDevice &d_csr,
                              const CSRHost &h_csr,
                              const float *d_x,
                              float *d_y);

// Runs exactly one kernel launch (LINE or FLAT, per the setup-time
// choice). Does not reset d_y and does not time itself.
void spmv_acc_run_once(AccSpMVContext &ctx);

// Frees FLAT's d_bp if it was allocated (no-op for LINE).
void spmv_acc_teardown(AccSpMVContext &ctx);

const char *spmv_acc_kernel_name(const AccSpMVContext &ctx);

// Per-call byte-accounting formula for whichever kernel was chosen (same
// formulas D1 used per kernel). FLAT's one-time bp[]-construction bytes are
// NOT included here, since that cost now lives in setup, not in the timed
// per-rep call.
long long spmv_acc_bytes_moved(const AccSpMVContext &ctx);
