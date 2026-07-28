#pragma once
#include <cuda_runtime.h>
#include <cusparse.h>
#include "matrix.hpp"

// Single-shot cuSPARSE SpMV.
struct CusparseSpMVContext
{
    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void *d_buffer;
    size_t buffer_size;
};

// One-time setup.
CusparseSpMVContext spmv_cusparse_setup(const CSRDevice &d_csr,
                                        const float *d_x,
                                        float *d_y);

// Runs exactly one cusparseSpMV call. Does not reset d_y (beta=0 already
// overwrites it rather than accumulating) and does not time itself.
void spmv_cusparse_run_once(CusparseSpMVContext &ctx);

void spmv_cusparse_teardown(CusparseSpMVContext &ctx);

// Byte-accounting formula (values + col_idx + row_ptr + x + y) exposed so
// callers can compute effective bandwidth from their own externally-timed
// avg_ms.
long long spmv_cusparse_bytes_moved(const CSRDevice &d_csr);
