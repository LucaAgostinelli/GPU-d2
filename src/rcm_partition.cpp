#include "rcm_partition.hpp"

#include <algorithm>
#include <numeric>
#include <queue>

RcmPartition compute_rcm_partition(const std::vector<COO> &coo, int n,
                                    int root, MPI_Comm comm)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    // `n` is only meaningful as passed on `root` (mirrors distribute_matrix()'s
    // own pattern) -- broadcast it so non-root ranks size owner_of/local_of
    // correctly before receiving them below.
    MPI_Bcast(&n, 1, MPI_INT, root, comm);

    RcmPartition rp;
    rp.owner_of.resize(n);
    rp.local_of.resize(n);
    rp.block_start.assign(P + 1, 0);

    if (rank == root)
    {
        // 1) Symmetrized undirected adjacency, flattened CSR-style (two-pass
        //    counting, no vector<vector<int>>). Self-loops (diagonal entries)
        //    contribute no edges.
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

        // Dedup neighbors per vertex in place (COO may carry duplicate
        // entries); `deg[v]` is overwritten with the dedup'd degree, which is
        // what the BFS below actually iterates over.
        for (int v = 0; v < n; v++)
        {
            int s = adj_start[v], e = adj_start[v + 1];
            std::sort(adj.begin() + s, adj.begin() + e);
            auto last = std::unique(adj.begin() + s, adj.begin() + e);
            deg[v] = (int)(last - (adj.begin() + s));
        }

        // 2) Cuthill-McKee: BFS ordering vertices, visiting each node's
        //    unvisited neighbors in ascending-degree order. New connected
        //    components (or isolated/zero-degree rows) start from the
        //    lowest-degree unvisited vertex -- a cheap proxy for a true
        //    pseudo-peripheral-vertex search.
        std::vector<int> by_degree(n);
        std::iota(by_degree.begin(), by_degree.end(), 0);
        std::sort(by_degree.begin(), by_degree.end(),
                  [&](int a, int b) { return deg[a] < deg[b]; });

        std::vector<char> visited(n, 0);
        std::vector<int> cm_order;
        cm_order.reserve(n);
        size_t degree_ptr = 0;
        std::vector<int> neigh_buf;

        for (;;)
        {
            while (degree_ptr < by_degree.size() && visited[by_degree[degree_ptr]])
                degree_ptr++;
            if (degree_ptr >= by_degree.size())
                break;

            int start = by_degree[degree_ptr];
            visited[start] = 1;
            cm_order.push_back(start);
            std::queue<int> q;
            q.push(start);

            while (!q.empty())
            {
                int u = q.front();
                q.pop();

                neigh_buf.clear();
                for (int idx = adj_start[u]; idx < adj_start[u] + deg[u]; idx++)
                {
                    int v = adj[idx];
                    if (!visited[v])
                        neigh_buf.push_back(v);
                }
                std::sort(neigh_buf.begin(), neigh_buf.end(),
                          [&](int a, int b) { return deg[a] < deg[b]; });

                for (int v : neigh_buf)
                {
                    if (visited[v])
                        continue;
                    visited[v] = 1;
                    cm_order.push_back(v);
                    q.push(v);
                }
            }
        }

        // 3) Reverse CM -> RCM.
        rp.order.assign(cm_order.rbegin(), cm_order.rend());

        // 4) Split the RCM order into P contiguous, NNZ-balanced blocks.
        long long total_nnz = 0;
        for (int c : nnz_per_row)
            total_nnz += c;

        long long target = total_nnz / P;
        long long acc = 0;
        int pos = 0;
        for (int r = 0; r < P - 1; r++)
        {
            long long want = target * (r + 1);
            while (pos < n && acc < want)
            {
                acc += nnz_per_row[rp.order[pos]];
                pos++;
            }
            // Guarantee a non-empty block even if a single dominant row's
            // NNZ already exceeds this block's whole target (e.g. one very
            // long row) -- always advance by at least one row when rows
            // remain, so no rank silently gets an empty local CSR.
            if (pos == rp.block_start[r] && pos < n)
            {
                acc += nnz_per_row[rp.order[pos]];
                pos++;
            }
            rp.block_start[r + 1] = pos;
        }
        rp.block_start[P] = n;

        // 5) owner_of / local_of from the block assignment.
        for (int r = 0; r < P; r++)
        {
            for (int k = rp.block_start[r]; k < rp.block_start[r + 1]; k++)
            {
                int g = rp.order[k];
                rp.owner_of[g] = r;
                rp.local_of[g] = k - rp.block_start[r];
            }
        }
    }

    MPI_Bcast(rp.owner_of.data(), n, MPI_INT, root, comm);
    MPI_Bcast(rp.local_of.data(), n, MPI_INT, root, comm);
    MPI_Bcast(rp.block_start.data(), P + 1, MPI_INT, root, comm);

    return rp;
}
