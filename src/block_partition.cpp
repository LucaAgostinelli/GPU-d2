#include "block_partition.hpp"

BlockPartition compute_block_partition(const std::vector<COO> &coo, int n,
                                        int root, MPI_Comm comm)
{
    (void)coo; // unused -- see header comment: pure arithmetic on n/P alone

    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // `n` is only meaningful as passed on `root` (mirrors the other two
    // prototypes' own pattern).
    MPI_Bcast(&n, 1, MPI_INT, root, comm);

    BlockPartition rp;

    // Same "as-even-as-possible" split as local_nrows() in
    // distribute_matrix.hpp (n/P rows per rank, +1 for the first n%P ranks)
    // -- identical per-rank ROW COUNT to the baseline cyclic driver, just
    // contiguous instead of interleaved.
    rp.block_start.assign(P + 1, 0);
    for (int r = 0; r < P; r++)
        rp.block_start[r + 1] = rp.block_start[r] + (n / P) + (r < n % P ? 1 : 0);

    // No root-only computation, no broadcast needed beyond `n` above --
    // every rank derives the identical owner_of/local_of/order directly.
    rp.owner_of.resize(n);
    rp.local_of.resize(n);
    rp.order.resize(n);
    for (int r = 0; r < P; r++)
    {
        for (int k = rp.block_start[r]; k < rp.block_start[r + 1]; k++)
        {
            rp.owner_of[k] = r;
            rp.local_of[k] = k - rp.block_start[r];
            rp.order[k] = k; // identity -- no reordering
        }
    }

    return rp;
}
