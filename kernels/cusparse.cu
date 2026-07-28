#include "cusparse.hpp"
#include <cstdio>
#include <cstdlib>

#define CUSPARSE_CHECK(call)                                \
    do                                                      \
    {                                                        \
        cusparseStatus_t _st = (call);                        \
        if (_st != CUSPARSE_STATUS_SUCCESS)                    \
        {                                                        \
            fprintf(stderr, "cuSPARSE error %d at %s:%d\n",       \
                    (int)_st, __FILE__, __LINE__);                 \
            exit(EXIT_FAILURE);                                      \
        }                                                              \
    } while (0)

#define CUDA_CHECK_LOCAL(call)                                  \
    do                                                          \
    {                                                            \
        cudaError_t _err = (call);                               \
        if (_err != cudaSuccess)                                  \
        {                                                          \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",           \
                    cudaGetErrorString(_err), __FILE__, __LINE__);  \
            exit(EXIT_FAILURE);                                     \
        }                                                            \
    } while (0)

CusparseSpMVContext spmv_cusparse_setup(const CSRDevice &d_csr,
                                        const float *d_x,
                                        float *d_y)
{
    CusparseSpMVContext ctx{};
    const int nrows = d_csr.nrows;
    const int ncols = d_csr.ncols;
    const int nnz = d_csr.nnz;

    CUSPARSE_CHECK(cusparseCreate(&ctx.handle));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &ctx.matA,
        nrows, ncols, nnz,
        (void *)d_csr.row_ptr,
        (void *)d_csr.col_idx,
        (void *)d_csr.values,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F));

    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vecX, ncols, (void *)d_x, CUDA_R_32F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&ctx.vecY, nrows, (void *)d_y, CUDA_R_32F));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    ctx.buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        ctx.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, ctx.matA, ctx.vecX,
        &beta, ctx.vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        &ctx.buffer_size));

    ctx.d_buffer = nullptr;
    if (ctx.buffer_size > 0)
        CUDA_CHECK_LOCAL(cudaMalloc(&ctx.d_buffer, ctx.buffer_size));

    return ctx;
}

void spmv_cusparse_run_once(CusparseSpMVContext &ctx)
{
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUSPARSE_CHECK(cusparseSpMV(
        ctx.handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, ctx.matA, ctx.vecX,
        &beta, ctx.vecY,
        CUDA_R_32F,
        CUSPARSE_SPMV_ALG_DEFAULT,
        ctx.d_buffer));
}

void spmv_cusparse_teardown(CusparseSpMVContext &ctx)
{
    if (ctx.d_buffer)
        CUDA_CHECK_LOCAL(cudaFree(ctx.d_buffer));
    CUSPARSE_CHECK(cusparseDestroySpMat(ctx.matA));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx.vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(ctx.vecY));
    CUSPARSE_CHECK(cusparseDestroy(ctx.handle));
}

long long spmv_cusparse_bytes_moved(const CSRDevice &d_csr)
{
    return (long long)d_csr.nnz * sizeof(float)          // values[]
         + (long long)d_csr.nnz * sizeof(int)             // col_idx[]
         + (long long)(d_csr.nrows + 1) * sizeof(int)      // row_ptr[]
         + (long long)d_csr.ncols * sizeof(float)           // x[]
         + (long long)d_csr.nrows * sizeof(float);           // y[]
}
