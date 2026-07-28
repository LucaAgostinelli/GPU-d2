#pragma once
#include <vector>
#include <string>

// Sparse matrix formats shared by every driver: COO (as read from disk) and
// CSR (the format every SpMV kernel in this project consumes).

struct COO
{
    int row, col;
    float val;
};

struct CSRHost
{
    int nrows, ncols, nnz;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

struct CSRDevice
{
    int nrows, ncols, nnz;
    int *row_ptr;
    int *col_idx;
    float *values;
};

// Reads a Matrix Market coordinate file into unsorted COO triplets
// (0-based indices).
std::vector<COO> read_mtx(const std::string &filename,
                          int &nrows, int &ncols,
                          bool &symmetric);

// Sorts `coo` in place by (row, col) and builds a CSR from it.
CSRHost coo_to_csr(std::vector<COO> &coo, int nrows, int ncols);

// Allocates device buffers and copies a host CSR onto the GPU.
CSRDevice csr_host_to_device(const CSRHost &h_csr);
