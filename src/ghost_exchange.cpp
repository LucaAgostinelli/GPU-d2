#include "ghost_exchange.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <unordered_set>

GhostPlan build_ghost_plan(LocalCSR &local, MPI_Comm comm)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    if (local.nrows_global != local.ncols_global)
    {
        fprintf(stderr,
                "build_ghost_plan: matrix must be square (got %d x %d) -- "
                "ghost ownership assumes x follows the same cyclic rule as rows.\n",
                local.nrows_global, local.ncols_global);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    // 1) Scan col_idx once, dedup remote columns per owning peer rank.
    //    owner(g) = g % P, that peer's own local index for g is g / P.
    std::vector<std::unordered_set<int>> remote_set(P);
    for (int g : local.col_idx)
    {
        int owner = g % P;
        if (owner != rank)
            remote_set[owner].insert(g / P);
    }

    std::vector<std::vector<int>> need_idx(P);
    for (int p = 0; p < P; p++)
    {
        if (remote_set[p].empty())
            continue;
        need_idx[p].assign(remote_set[p].begin(), remote_set[p].end());
        std::sort(need_idx[p].begin(), need_idx[p].end());
    }
    remote_set.clear();

    GhostPlan plan;
    plan.need_counts.assign(P, 0);
    plan.need_displs.assign(P, 0);
    int ghost_off = 0;
    for (int p = 0; p < P; p++)
    {
        plan.need_counts[p] = (int)need_idx[p].size();
        plan.need_displs[p] = ghost_off;
        ghost_off += plan.need_counts[p];
    }
    plan.n_ghost = ghost_off;

    // 2) Translate col_idx GLOBAL -> compact local space:
    //      owned column g  -> g / P                          (in [0, nrows_local))
    //      remote column g -> nrows_local + ghost slot        (in [nrows_local, nrows_local+n_ghost))
    //    Ghost slot found via binary search in the peer's sorted need_idx (built above,
    //    same order used to lay out this rank's ghost region).
    for (int &g : local.col_idx)
    {
        int owner = g % P;
        int lidx  = g / P;
        if (owner == rank)
        {
            g = lidx;
        }
        else
        {
            const auto &v = need_idx[owner];
            int pos = (int)(std::lower_bound(v.begin(), v.end(), lidx) - v.begin());
            g = local.nrows_local + plan.need_displs[owner] + pos;
        }
    }

    // 3) Count exchange: my "need_counts[p]" (how many values I need FROM p) fed into
    //    one Alltoall. By the transpose property, my resulting recvcount for peer p
    //    equals p's sendcount for me -- i.e. how many values p NEEDS FROM ME.
    plan.req_counts.assign(P, 0);
    MPI_Alltoall(plan.need_counts.data(), 1, MPI_INT,
                 plan.req_counts.data(), 1, MPI_INT, comm);

    plan.req_displs.assign(P, 0);
    int req_total = 0;
    for (int p = 0; p < P; p++)
    {
        plan.req_displs[p] = req_total;
        req_total += plan.req_counts[p];
    }
    plan.requested_of_me.resize(req_total);

    // 4) Index-list exchange: send my need_idx (flattened, peer-ordered), receive,
    //    for each peer, the list of MY OWN local indices that peer needs from me --
    //    exactly the gather list for the runtime value exchange.
    std::vector<int> flat_need(ghost_off);
    for (int p = 0; p < P; p++)
        std::copy(need_idx[p].begin(), need_idx[p].end(), flat_need.begin() + plan.need_displs[p]);

    MPI_Alltoallv(flat_need.data(), plan.need_counts.data(), plan.need_displs.data(), MPI_INT,
                  plan.requested_of_me.data(), plan.req_counts.data(), plan.req_displs.data(), MPI_INT,
                  comm);

    return plan;
}
