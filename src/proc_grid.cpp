#include "proc_grid.hpp"

#include <cmath>

ProcGrid make_proc_grid(MPI_Comm comm)
{
    ProcGrid g;
    MPI_Comm_rank(comm, &g.rank);
    MPI_Comm_size(comm, &g.P);

    int P_r = (int)std::sqrt((double)g.P);
    if (P_r < 1)
        P_r = 1;
    while (P_r > 1 && g.P % P_r != 0)
        P_r--;

    g.P_r = P_r;
    g.P_c = g.P / P_r;

    g.pr = g.rank / g.P_c;
    g.pc = g.rank % g.P_c;

    // color = group, key = rank-within-group (determines ordering).
    MPI_Comm_split(comm, g.pr, g.pc, &g.row_comm);
    MPI_Comm_split(comm, g.pc, g.pr, &g.col_comm);

    return g;
}

void free_proc_grid(ProcGrid &grid)
{
    MPI_Comm_free(&grid.row_comm);
    MPI_Comm_free(&grid.col_comm);
}
