/**
 * 2D checkerboard partitioning of a sparse matrix over a P_r x P_c process
 * grid (see proc_grid.hpp).
 *
 *   owner(i,j)   = (i mod P_r, j mod P_c)
 *   local_row(i) = i / P_r        local_col(j) = j / P_c
 *
 * Unlike 1D's distribute_matrix() (which leaves col_idx in GLOBAL space for
 * ghost_exchange.hpp to translate later), col_idx here is already COMPACT
 * LOCAL ([0, ncols_local)) at distribution time -- there is no ghost
 * concept in 2D. Every column a rank's block references belongs to the
 * SAME fixed set of P_r peers (the ones sharing its pc): the whole local
 * column range is delivered by one MPI_Bcast (or ncclBroadcast) within the
 * column communicator, not a per-index sparse exchange.
 *
 * Requires a SQUARE matrix (nrows_global == ncols_global), same requirement
 * as the 1D ghost exchange and for the same reason: columns must follow the
 * same cyclic rule as rows for the row/column process-grid axes to line up
 * with a single shared dense vector x.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "matrix.hpp"
#include "proc_grid.hpp"

struct LocalCSR2D
{
    int rank, P;
    int P_r, P_c, pr, pc;
    int nrows_global, ncols_global;
    int nrows_local, ncols_local, nnz_local;

    std::vector<int>   row_ptr;  // size nrows_local + 1, LOCAL row indexing
    std::vector<int>   col_idx;  // size nnz_local, LOCAL column indexing [0, ncols_local)
    std::vector<float> values;   // size nnz_local
};

// Collective call: every rank in `comm` must call this together, passing
// the SAME ProcGrid every rank already built via make_proc_grid(comm).
//
// On `root`, `global_coo`/`nrows_global`/`ncols_global` must hold the full
// matrix; `global_coo` is bucketed and cleared as a side effect. On
// non-root ranks these three arguments are don't-care inputs -- true
// dimensions are broadcast internally.
LocalCSR2D distribute_matrix_2d(std::vector<COO> &global_coo,
                                 int nrows_global,
                                 int ncols_global,
                                 const ProcGrid &grid,
                                 int root,
                                 MPI_Comm comm);
