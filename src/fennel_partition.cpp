#include "fennel_partition.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

// LDG's load-balance penalty is a SOFT constraint (see compute_fennel_partition's
// scoring loop): a partition already at `capacity` is discouraged, not forbidden,
// from taking more. 10% slack over the exact nnz/P average keeps the penalty
// informative near the target instead of clamping hard exactly at it.
static const double FENNEL_CAPACITY_SLACK = 1.10;

FennelPartition compute_fennel_partition(const std::vector<COO> &coo, int n,
                                          int root, MPI_Comm comm)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // Mirrors compute_rcm_partition()'s own pattern: `n` is only meaningful
    // as passed on `root`, broadcast it before sizing owner_of/local_of.
    MPI_Bcast(&n, 1, MPI_INT, root, comm);

    FennelPartition rp;
    rp.owner_of.assign(n, -1);
    rp.local_of.assign(n, 0);
    rp.block_start.assign(P + 1, 0);

    if (rank == root)
    {
        // 1) Symmetrized undirected adjacency, flattened CSR-style (two-pass
        //    counting, dedup per vertex).
        std::vector<int> nnz_per_row(n, 0);
        std::vector<int> deg(n, 0);
        for (const COO &e : coo)
        {
            nnz_per_row[e.row]++;
            if (e.row != e.col)
            {
                deg[e.row]++;
                deg[e.col]++;
            }
        }

        std::vector<int> adj_start(n + 1, 0);
        for (int i = 0; i < n; i++)
            adj_start[i + 1] = adj_start[i] + deg[i];

        std::vector<int> adj(adj_start[n]);
        {
            std::vector<int> cursor(adj_start.begin(), adj_start.end() - 1);
            for (const COO &e : coo)
            {
                if (e.row == e.col)
                    continue;
                adj[cursor[e.row]++] = e.col;
                adj[cursor[e.col]++] = e.row;
            }
        }

        for (int v = 0; v < n; v++)
        {
            int s = adj_start[v], e = adj_start[v + 1];
            std::sort(adj.begin() + s, adj.begin() + e);
            auto last = std::unique(adj.begin() + s, adj.begin() + e);
            deg[v] = (int)(last - (adj.begin() + s));
        }

        // 2) Processing order: descending degree. Hub vertices are placed
        //    first, while every partition's load is still small and
        //    comparable, giving the hardest-to-place vertices the most
        //    freedom instead of whatever capacity remains last.
        std::vector<int> order_by_degree(n);
        std::iota(order_by_degree.begin(), order_by_degree.end(), 0);
        std::sort(order_by_degree.begin(), order_by_degree.end(),
                  [&](int a, int b) { return deg[a] > deg[b]; });

        long long total_nnz = 0;
        for (int c : nnz_per_row)
            total_nnz += c;
        double capacity = (double)total_nnz / P * FENNEL_CAPACITY_SLACK;
        if (capacity < 1.0)
            capacity = 1.0;

        std::vector<long long> load(P, 0);
        std::vector<std::vector<int>> local_to_global(P);
        std::vector<int> neighbor_count(P);

        for (int v : order_by_degree)
        {
            std::fill(neighbor_count.begin(), neighbor_count.end(), 0);
            for (int idx = adj_start[v]; idx < adj_start[v] + deg[v]; idx++)
            {
                int owner = rp.owner_of[adj[idx]];
                if (owner >= 0)
                    neighbor_count[owner]++;
            }

            // argmax_p [ neighbor_count[p] * (1 - load[p]/capacity) ] (LDG score);
            // ties (all-zero neighbor overlap, e.g. every hub's first vertex, or
            // genuinely equal scores) broken by smallest current load -- this
            // round-robins the earliest vertices across all P partitions instead
            // of always favoring partition 0.
            int best_p = 0;
            double best_score = -1e18;
            for (int p = 0; p < P; p++)
            {
                double score = neighbor_count[p] * (1.0 - (double)load[p] / capacity);
                if (score > best_score ||
                    (score == best_score && load[p] < load[best_p]))
                {
                    best_score = score;
                    best_p = p;
                }
            }

            rp.owner_of[v] = best_p;
            rp.local_of[v] = (int)local_to_global[best_p].size();
            local_to_global[best_p].push_back(v);
            load[best_p] += nnz_per_row[v];
        }

        // 3) Flatten the P per-partition vertex lists into order[]/block_start[],
        //    the exact same shape RcmPartition produces -- so every downstream
        //    consumer (distribute_matrix_partitioned, ghost_exchange_mapped, the
        //    driver's x-scatter/y-gather bookkeeping) is reused unchanged.
        rp.order.resize(n);
        int off = 0;
        for (int p = 0; p < P; p++)
        {
            rp.block_start[p] = off;
            for (int v : local_to_global[p])
                rp.order[off++] = v;
        }
        rp.block_start[P] = off; // == n
    }

    MPI_Bcast(rp.owner_of.data(), n, MPI_INT, root, comm);
    MPI_Bcast(rp.local_of.data(), n, MPI_INT, root, comm);
    MPI_Bcast(rp.block_start.data(), P + 1, MPI_INT, root, comm);

    return rp;
}
