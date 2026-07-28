#include "matrix.hpp"
#include <cuda_runtime.h>

CSRDevice csr_host_to_device(const CSRHost &h_csr)
{
    CSRDevice d_csr;
    d_csr.nrows = h_csr.nrows;
    d_csr.ncols = h_csr.ncols;
    d_csr.nnz = h_csr.nnz;

    cudaMalloc(&d_csr.row_ptr, (h_csr.nrows + 1) * sizeof(int));
    cudaMalloc(&d_csr.col_idx, h_csr.nnz * sizeof(int));
    cudaMalloc(&d_csr.values, h_csr.nnz * sizeof(float));

    cudaMemcpy(d_csr.row_ptr, h_csr.row_ptr.data(),
               (h_csr.nrows + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.col_idx, h_csr.col_idx.data(),
               h_csr.nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csr.values, h_csr.values.data(),
               h_csr.nnz * sizeof(float), cudaMemcpyHostToDevice);

    return d_csr;
}
