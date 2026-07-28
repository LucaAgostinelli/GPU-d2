/**
 * Locality-aware 1D row/column partitioning: Reverse Cuthill-McKee (RCM)
 * reordering of the matrix's sparsity-pattern graph, then split into P
 * contiguous, NNZ-balanced blocks over that reordering. Clustering mutually-adjacent
 * rows/columns together before slicing keeps most of a block's nonzero
 * references inside the same block, shrinking the ghost set relative to
 * cyclic on locality-rich matrices.
 *
 * Opt-in path only, consumed by spmv_rcm_* drivers via
 * distribute_matrix_partitioned.hpp / ghost_exchange_mapped.hpp -- does not
 * touch distribute_matrix.hpp/ghost_exchange.hpp.
 *
 * Same square-matrix assumption as ghost_exchange.hpp: column ownership
 * follows row ownership (owner_of/local_of apply to both), treating the
 * sparsity pattern as a single undirected graph over row==column vertices.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "matrix.hpp"

struct RcmPartition
{
    std::vector<int> owner_of;    // size n, valid on ALL ranks (broadcast)
    std::vector<int> local_of;    // size n, valid on ALL ranks (broadcast)
    std::vector<int> block_start; // size P+1, valid on ALL ranks (broadcast) --
                                   // rank r owns order[block_start[r] : block_start[r+1])
    std::vector<int> order;       // size n, ROOT ONLY -- order[k] = original row id
                                   // at RCM+NNZ-balanced position k. Used by the
                                   // caller on root for x-scatter setup and
                                   // y-gather reconstruction (the local-index ->
                                   // global-id inverse mapping).
};

// Collective call: every rank in `comm` must call this together.
//
// On `root`, `coo` must hold the FULL matrix and is READ-ONLY here (unlike
// distribute_matrix_partitioned(), called afterward, which DOES bucket/clear
// it) -- the caller still needs `coo` intact for that next call. `n` is
// nrows_global (== ncols_global, matrix must be square). On non-root ranks
// `coo`/`n` are don't-care inputs.
RcmPartition compute_rcm_partition(const std::vector<COO> &coo, int n,
                                    int root, MPI_Comm comm);
