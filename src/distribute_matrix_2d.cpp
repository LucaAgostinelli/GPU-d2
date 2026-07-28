#include "distribute_matrix_2d.hpp"

#include <cstdio>
#include <cstdlib>

#include "distribute_matrix.hpp" // local_nrows() -- reused as-is, parametrized by P_r/P_c below

LocalCSR2D distribute_matrix_2d(std::vector<COO> &global_coo,
                                 int nrows_global, int ncols_global,
                                 const ProcGrid &grid,
                                 int root, MPI_Comm comm)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    MPI_Bcast(&nrows_global, 1, MPI_INT, root, comm);
    MPI_Bcast(&ncols_global, 1, MPI_INT, root, comm);

    if (nrows_global != ncols_global)
    {
        fprintf(stderr,
                "distribute_matrix_2d: matrix must be square (got %d x %d) -- "
                "2D checkerboard partitioning assumes x follows the same cyclic "
                "rule on columns as rows do.\n",
                nrows_global, ncols_global);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // ---- Buffers only meaningful on root ----
    std::vector<int>   nnz_counts;
    std::vector<int>   rp_sendcounts, rp_displs;
    std::vector<int>   ci_sendcounts, ci_displs;
    std::vector<int>   flat_row_ptr;
    std::vector<int>   flat_col_idx;
    std::vector<float> flat_values;

    if (rank == root)
    {
        // 1) Bucket by owner(i,j) = (i mod P_r, j mod P_c) -> rank = pr*P_c+pc,
        //    rewriting (row,col) to LOCAL (row/P_r, col/P_c).
        std::vector<std::vector<COO>> buckets(P);
        for (auto &b : buckets)
            b.reserve(global_coo.size() / P + 1);

        for (const COO &e : global_coo)
        {
            int br = e.row % grid.P_r;
            int bc = e.col % grid.P_c;
            int owner = br * grid.P_c + bc;
            int local_row = e.row / grid.P_r;
            int local_col = e.col / grid.P_c;
            buckets[owner].push_back(COO{local_row, local_col, e.val});
        }
        global_coo.clear();
        global_coo.shrink_to_fit();

        // 2) Per-bucket COO -> CSR. Each bucket's local dims come from its
        //    own (pr,pc), not from a single shared P like 1D.
        std::vector<CSRHost> local_csrs(P);
        nnz_counts.resize(P);
        for (int r = 0; r < P; r++)
        {
            int pr_r = r / grid.P_c;
            int pc_r = r % grid.P_c;
            int nrows_r = local_nrows(nrows_global, grid.P_r, pr_r);
            int ncols_r = local_nrows(ncols_global, grid.P_c, pc_r);
            local_csrs[r] = coo_to_csr(buckets[r], nrows_r, ncols_r);
            nnz_counts[r] = local_csrs[r].nnz;
        }

        // 3) Flatten for Scatterv (identical shape to distribute_matrix.cpp).
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

    LocalCSR2D out;
    out.rank = rank; out.P = P;
    out.P_r = grid.P_r; out.P_c = grid.P_c; out.pr = grid.pr; out.pc = grid.pc;
    out.nrows_global = nrows_global; out.ncols_global = ncols_global;
    out.nrows_local = local_nrows(nrows_global, grid.P_r, grid.pr);
    out.ncols_local = local_nrows(ncols_global, grid.P_c, grid.pc);

    // 4) nnz_local is data-dependent -- must be communicated before the
    //    Scatterv calls below can size their receive buffers.
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
