#include "distribute_matrix_partitioned.hpp"

LocalCSR distribute_matrix_partitioned(std::vector<COO> &global_coo,
                                        int nrows_global, int ncols_global,
                                        const std::vector<int> &owner_of,
                                        const std::vector<int> &local_of,
                                        const std::vector<int> &block_start,
                                        int root, MPI_Comm comm)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    MPI_Bcast(&nrows_global, 1, MPI_INT, root, comm);
    MPI_Bcast(&ncols_global, 1, MPI_INT, root, comm);

    std::vector<int>   nnz_counts;
    std::vector<int>   rp_sendcounts, rp_displs;
    std::vector<int>   ci_sendcounts, ci_displs;
    std::vector<int>   flat_row_ptr;
    std::vector<int>   flat_col_idx;
    std::vector<float> flat_values;

    if (rank == root)
    {
        // 1) Bucket by owner_of(row)/local_of(row), instead of row % P / row / P.
        std::vector<std::vector<COO>> buckets(P);
        for (auto &b : buckets)
            b.reserve(global_coo.size() / P + 1);

        for (const COO &e : global_coo)
        {
            int owner = owner_of[e.row];
            int local_row = local_of[e.row];
            buckets[owner].push_back(COO{local_row, e.col, e.val});
        }
        global_coo.clear();
        global_coo.shrink_to_fit();

        // 2) Per-bucket COO -> CSR.
        std::vector<CSRHost> local_csrs(P);
        nnz_counts.resize(P);
        for (int r = 0; r < P; r++)
        {
            int nrows_r = block_start[r + 1] - block_start[r];
            local_csrs[r] = coo_to_csr(buckets[r], nrows_r, ncols_global);
            nnz_counts[r] = local_csrs[r].nnz;
        }

        // 3) Flatten for Scatterv.
        rp_sendcounts.resize(P); rp_displs.resize(P);
        ci_sendcounts.resize(P); ci_displs.resize(P);

        int rp_off = 0, ci_off = 0;
        for (int r = 0; r < P; r++)
        {
            int nrows_r = local_csrs[r].nrows;

            rp_sendcounts[r] = nrows_r + 1;
            rp_displs[r]     = rp_off;
            flat_row_ptr.insert(flat_row_ptr.end(),
                                 local_csrs[r].row_ptr.begin(), local_csrs[r].row_ptr.end());
            rp_off += nrows_r + 1;

            ci_sendcounts[r] = local_csrs[r].nnz;
            ci_displs[r]     = ci_off;
            flat_col_idx.insert(flat_col_idx.end(),
                                 local_csrs[r].col_idx.begin(), local_csrs[r].col_idx.end());
            flat_values.insert(flat_values.end(),
                                local_csrs[r].values.begin(), local_csrs[r].values.end());
            ci_off += local_csrs[r].nnz;
        }
    }

    LocalCSR out;
    out.rank = rank; out.P = P;
    out.nrows_global = nrows_global; out.ncols_global = ncols_global;
    out.nrows_local  = block_start[rank + 1] - block_start[rank];

    MPI_Scatter(rank == root ? nnz_counts.data() : nullptr, 1, MPI_INT,
                &out.nnz_local, 1, MPI_INT, root, comm);

    out.row_ptr.resize(out.nrows_local + 1);
    out.col_idx.resize(out.nnz_local);
    out.values.resize(out.nnz_local);

    MPI_Scatterv(rank == root ? flat_row_ptr.data() : nullptr,
                 rank == root ? rp_sendcounts.data() : nullptr,
                 rank == root ? rp_displs.data()     : nullptr,
                 MPI_INT,
                 out.row_ptr.data(), out.nrows_local + 1, MPI_INT,
                 root, comm);

    MPI_Scatterv(rank == root ? flat_col_idx.data() : nullptr,
                 rank == root ? ci_sendcounts.data() : nullptr,
                 rank == root ? ci_displs.data()     : nullptr,
                 MPI_INT,
                 out.col_idx.data(), out.nnz_local, MPI_INT,
                 root, comm);

    MPI_Scatterv(rank == root ? flat_values.data() : nullptr,
                 rank == root ? ci_sendcounts.data() : nullptr,
                 rank == root ? ci_displs.data()     : nullptr,
                 MPI_FLOAT,
                 out.values.data(), out.nnz_local, MPI_FLOAT,
                 root, comm);

    return out;
}
