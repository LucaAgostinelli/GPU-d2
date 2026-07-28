/**
 * 1D cyclic (modulo) row distribution of a sparse matrix over MPI ranks.
 *
 * owner(i) = i mod P        (i = global row index, P = communicator size)
 * local(i) = i / P          (row's index within its owner's local CSR)
 *
 * Column indices are left in GLOBAL space here -- ghost_exchange.hpp is what
 * localizes columns and figures out which remote x entries are needed.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "matrix.hpp"

// Local CSR shard owned by one rank after 1D cyclic distribution.
struct LocalCSR
{
    int rank, P;
    int nrows_global, ncols_global;
    int nrows_local, nnz_local;

    std::vector<int>   row_ptr;  // size nrows_local + 1, LOCAL row indexing
    std::vector<int>   col_idx;  // size nnz_local, GLOBAL column indices
    std::vector<float> values;   // size nnz_local
};

// Number of rows owned by `rank` under owner(i) = i % P, for a matrix with
// `nrows_global` rows split over `P` ranks. Pure arithmetic -- every rank
// can compute this independently, no communication needed.
int local_nrows(int nrows_global, int P, int rank);

// Collective call: every rank in `comm` must call this together.
//
// On `root`, `global_coo`/`nrows_global`/`ncols_global` must hold the full
// matrix; `global_coo` is bucketed and cleared as a side effect (copy it
// first if the caller still needs the original afterwards). On non-root
// ranks these three arguments are don't-care inputs -- the true dimensions
// are broadcast internally, so every rank returns a correctly-sized result.
LocalCSR distribute_matrix(std::vector<COO> &global_coo,
                            int nrows_global,
                            int ncols_global,
                            int root,
                            MPI_Comm comm);
