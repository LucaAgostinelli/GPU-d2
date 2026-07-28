/**
 * Generalization of ghost_exchange.hpp's build_ghost_plan() to an ARBITRARY
 * row/column ownership (owner_of[g]/local_of[g], as produced by e.g.
 * compute_rcm_partition()) instead of the hardcoded owner(g) = g % P /
 * local_idx(g) = g / P cyclic rule.
 *
 * Reuses the GhostPlan struct from ghost_exchange.hpp unchanged; the output
 * shape and runtime exchange protocol are identical, only the column ->
 * (owner, local index) lookup changes from arithmetic to array lookup. Same
 * square-matrix assumption: owner_of/local_of apply uniformly to row and
 * column global ids.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "distribute_matrix.hpp" // LocalCSR
#include "ghost_exchange.hpp"    // GhostPlan

// Collective call: every rank in `comm` must call this together.
// Mutates local.col_idx in place (GLOBAL -> compact local space), exactly
// like build_ghost_plan(), but using owner_of[g]/local_of[g] lookups instead
// of g % P / g / P.
GhostPlan build_ghost_plan_mapped(LocalCSR &local,
                                   const std::vector<int> &owner_of,
                                   const std::vector<int> &local_of,
                                   MPI_Comm comm);
