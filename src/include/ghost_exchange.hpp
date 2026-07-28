/**
 * Selective ghost-entry exchange for 1D cyclic partitioning.
 *
 * Assumes a SQUARE matrix (nrows_global == ncols_global), so x follows the
 * exact same 1D cyclic rule as the rows:
 *   owner(col) = col % P          local_idx(col) = col / P
 *
 * build_ghost_plan() rewrites a LocalCSR's col_idx from GLOBAL column space
 * into a COMPACT LOCAL space:
 *   [0, nrows_local)                -- columns this rank owns, translated
 *                                       to col / P (aligned with this rank's
 *                                       own owned slice of x)
 *   [nrows_local, nrows_local+n_ghost) -- remote ("ghost") columns, one slot
 *                                       per UNIQUE remote column actually
 *                                       referenced, grouped contiguously by
 *                                       owning peer rank (peer 0, 1, ...)
 *
 * It also runs the one-time preprocessing exchange (index-request Alltoall +
 * Alltoallv) that tells every rank exactly which of its own x entries other
 * ranks need -- see GhostPlan::requested_of_me. That list is everything the
 * runtime value exchange (done once per driver, on device, GPU-aware MPI or
 * NCCL) needs to pack its send buffer.
 */
#pragma once

#include <vector>
#include <mpi.h>

#include "distribute_matrix.hpp" // LocalCSR

struct GhostPlan
{
    int n_ghost; // number of unique remote columns referenced locally

    // ---- Runtime VALUE exchange (repeated every driver run, GPU-aware) ----
    // "How many values I need FROM peer j" / "where in the recv buffer".
    // Recv buffer for this exchange is d_x_full + nrows_local, i.e. the
    // ghost region; slot order matches the col_idx translation above.
    std::vector<int> need_counts, need_displs;

    // "How many values peer j needs FROM me" / requested_of_me is the
    // gather list (indices into MY OWN owned x) to pack into the send
    // buffer for this exchange, in peer order.
    std::vector<int> req_counts, req_displs;
    std::vector<int> requested_of_me;
};

// Collective call: every rank in `comm` must call this together.
// Mutates local.col_idx in place (GLOBAL -> compact local space).
GhostPlan build_ghost_plan(LocalCSR &local, MPI_Comm comm);
