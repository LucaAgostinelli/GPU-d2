/**
 * Streaming greedy edge-cut partitioning (LDG -- Linear Deterministic Graph
 * partitioning): each vertex is assigned directly to
 * whichever partition already holds the most of its neighbors, weighted by
 * a load-balance penalty.
 *
 * Mirrors RcmPartition's output shape (owner_of/local_of/block_start/order)
 * so distribute_matrix_partitioned.hpp and ghost_exchange_mapped.hpp are
 * reused unchanged; only the partition-computation module differs.
 *
 * Same square-matrix assumption as rcm_partition.hpp/ghost_exchange.hpp:
 * owner_of/local_of apply uniformly to row and column global ids.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "matrix.hpp"

struct FennelPartition
{
    std::vector<int> owner_of;    // size n, valid on ALL ranks (broadcast)
    std::vector<int> local_of;    // size n, valid on ALL ranks (broadcast)
    std::vector<int> block_start; // size P+1, valid on ALL ranks (broadcast) --
                                   // rank r owns order[block_start[r] : block_start[r+1])
    std::vector<int> order;       // size n, ROOT ONLY -- order[k] = original row id
                                   // whose LOCAL index (on its assigned rank) is
                                   // k - block_start[owner]. Same role as
                                   // RcmPartition::order, used identically by the
                                   // caller for x-scatter setup and y-gather
                                   // reconstruction.
};

// Collective call: every rank in `comm` must call this together.
//
// On `root`, `coo` must hold the FULL matrix and is READ-ONLY here (same
// contract as compute_rcm_partition() -- the caller still needs `coo` intact
// for the distribute_matrix_partitioned() call that follows). `n` is
// nrows_global (== ncols_global, matrix must be square). On non-root ranks
// `coo`/`n` are don't-care inputs.
FennelPartition compute_fennel_partition(const std::vector<COO> &coo, int n,
                                          int root, MPI_Comm comm);
