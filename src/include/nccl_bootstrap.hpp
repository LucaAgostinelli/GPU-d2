#pragma once
#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <nccl.h>

#define NCCL_CHECK(call)                                          \
    do                                                            \
    {                                                              \
        ncclResult_t _res = (call);                                 \
        if (_res != ncclSuccess)                                     \
        {                                                              \
            printf("NCCL error: %s\n", ncclGetErrorString(_res));       \
            exit(1);                                                     \
        }                                                                 \
    } while (0)

// One-time NCCL communicator bootstrap over an arbitrary MPI communicator:
// rank 0 (within mpi_comm) generates a ncclUniqueId, broadcasts it to every
// other rank in mpi_comm, then every rank calls ncclCommInitRank with its
// rank/size WITHIN that communicator. Used both for a single world-sized
// communicator (1D ghost exchange) and for per-axis sub-communicators (2D
// checkerboard's row_comm/col_comm, called once per axis).
inline ncclComm_t bootstrap_nccl_comm(MPI_Comm mpi_comm)
{
    int rank, size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    ncclUniqueId id;
    if (rank == 0)
        NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

    ncclComm_t comm;
    NCCL_CHECK(ncclCommInitRank(&comm, size, id, rank));
    return comm;
}
