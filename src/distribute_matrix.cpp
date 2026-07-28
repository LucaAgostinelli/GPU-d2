#include "distribute_matrix.hpp"

int local_nrows(int nrows_global, int P, int rank)
{
    return nrows_global / P + (rank < nrows_global % P ? 1 : 0);
}

LocalCSR distribute_matrix(std::vector<COO> &global_coo,
                            int nrows_global, int ncols_global,
                            int root, MPI_Comm comm)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // Every rank ends up with the correct dims, regardless of what it passed in.
    MPI_Bcast(&nrows_global, 1, MPI_INT, root, comm);
    MPI_Bcast(&ncols_global, 1, MPI_INT, root, comm);

    // ---- Buffers only meaningful on root ----
    std::vector<int>   nnz_counts;
    std::vector<int>   rp_sendcounts, rp_displs;
    std::vector<int>   ci_sendcounts, ci_displs;
    std::vector<int>   flat_row_ptr;
    std::vector<int>   flat_col_idx;
    std::vector<float> flat_values;

    if (rank == root)
    {
        // 1) Bucket by owner(row) = row % P, rewriting row to LOCAL index (row / P).
        std::vector<std::vector<COO>> buckets(P);
        for (auto &b : buckets)
            b.reserve(global_coo.size() / P + 1);

        for (const COO &e : global_coo)
        {
            int owner = e.row % P;
            int local_row = e.row / P;
            buckets[owner].push_back(COO{local_row, e.col, e.val});
        }
        global_coo.clear();
        global_coo.shrink_to_fit();

        // 2) Per-bucket COO -> CSR (reuses coo_to_csr; sorts each bucket in
        //    place by (local_row, col) ascending -- matches the ordering
        //    coo_to_csr would give the same rows in the un-partitioned matrix,
        //    since local_row is monotonic in global row for a fixed owner).
        std::vector<CSRHost> local_csrs(P);
        nnz_counts.resize(P);
        for (int r = 0; r < P; r++)
        {
            int nrows_r = local_nrows(nrows_global, P, r);
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
    out.nrows_local  = local_nrows(nrows_global, P, rank);

    // 4) nnz_local is data-dependent -- must be communicated before the
    //    Scatterv calls below can size their receive buffers.
    MPI_Scatter(rank == root ? nnz_counts.data() : nullptr, 1, MPI_INT,
                &out.nnz_local, 1, MPI_INT, root, comm);

    out.row_ptr.resize(out.nrows_local + 1);
    out.col_idx.resize(out.nnz_local);
    out.values.resize(out.nnz_local);

    // 5) The three Scatterv calls. col_idx/values share sendcounts/displs
    //    since both are indexed by nnz.
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
