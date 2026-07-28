/**
 * Generalization of distribute_matrix.hpp to an ARBITRARY row/column
 * assignment (owner_of[g]/local_of[g]/block_start[], as produced by e.g.
 * compute_rcm_partition() in rcm_partition.hpp) instead of the baseline
 * owner(i) = i mod P cyclic rule.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "matrix.hpp"
#include "distribute_matrix.hpp" // reuses the LocalCSR struct

// Collective call: every rank in `comm` must call this together.
//
// On `root`, `global_coo` must hold the full matrix; it is bucketed and
// cleared as a side effect (same contract as distribute_matrix()). On
// non-root ranks it's a don't-care input. `owner_of`, `local_of`, and
// `block_start` must already be valid (broadcast) on every rank -- e.g. the
// output of compute_rcm_partition().
LocalCSR distribute_matrix_partitioned(std::vector<COO> &global_coo,
                                        int nrows_global, int ncols_global,
                                        const std::vector<int> &owner_of,
                                        const std::vector<int> &local_of,
                                        const std::vector<int> &block_start,
                                        int root, MPI_Comm comm);
