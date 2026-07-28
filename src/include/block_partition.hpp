/**
 * Classical "1D-Block" partitioning: contiguous row ranges over the
 * matrix's original row order, split as evenly as possible by row count
 * (same per-rank row count as the baseline cyclic driver's local_nrows(),
 * just contiguous instead of interleaved). No NNZ balancing, no reordering,
 * no view of the matrix's sparsity pattern needed -- pure arithmetic on n
 * and P, so every rank computes the identical result independently.
 *
 * Mirrors RcmPartition/FennelPartition's field shape so
 * distribute_matrix_partitioned.hpp and ghost_exchange_mapped.hpp are
 * reused unchanged.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "matrix.hpp"

struct BlockPartition
{
    std::vector<int> owner_of;    // size n, valid on ALL ranks
    std::vector<int> local_of;    // size n, valid on ALL ranks
    std::vector<int> block_start; // size P+1, valid on ALL ranks
    std::vector<int> order;       // size n, valid on ALL ranks -- order[k] == k (no reordering)
};

// Collective call: every rank in `comm` must call this together. `coo` is
// unused (kept only to match compute_rcm_partition()'s/
// compute_fennel_partition()'s signature). `n` = nrows_global, broadcast
// internally from `root`.
BlockPartition compute_block_partition(const std::vector<COO> &coo, int n,
                                        int root, MPI_Comm comm);
