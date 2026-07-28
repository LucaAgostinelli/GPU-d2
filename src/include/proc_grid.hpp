/**
 * P_r x P_c process grid for 2D checkerboard SpMV partitioning.
 *
 * Rank layout is row-major: rank = pr * P_c + pc. P_r is chosen as the
 * largest divisor of P that is <= sqrt(P) (so P_r <= P_c, and P_r == P_c
 * whenever P is a perfect square). For prime P (e.g. P=2, P=3) this
 * degenerates to a 1 x P grid -- still a correct, well-defined grid (every
 * rank shares the same pr, so row_comm has size 1 and the y-reduction step
 * is a no-op), just not a genuine checkerboard.
 */
#pragma once

#include <mpi.h>

struct ProcGrid
{
    int rank, P;
    int P_r, P_c;
    int pr, pc;         // this rank's grid coordinates

    MPI_Comm row_comm;  // fixed pr, varying pc -- size P_c, rank-in-comm == pc
    MPI_Comm col_comm;  // fixed pc, varying pr -- size P_r, rank-in-comm == pr
};

// Collective call: every rank in `comm` must call this together.
ProcGrid make_proc_grid(MPI_Comm comm);

// Frees row_comm/col_comm. Collective, same rules as MPI_Comm_free.
void free_proc_grid(ProcGrid &grid);
