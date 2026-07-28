#!/usr/bin/env python3
"""
weak-scaling summary for the three
intelligent-partitioning prototypes (RCM, Fennel/LDG, classical 1D-Block),
from bonus_partitioning_strategies/sbatch/weak_scaling_partitioned.sh.

mean_rank_effective_gflops (MEAN of each rank's own 2*nnz_local_i/T_i) and aggregate_gflops
(2*(sum of nnz_local across ranks)/T_max). Additionally cross-references the cyclic driver's own
weak-scaling numbers (comm=ghost_nccl, the best-performing 1D combination) at the same (kernel, P) from
../outputs/csv/weak_scaling_summary.csv, reporting speedup_vs_cyclic_ghost_nccl

NOTE: which cyclic summary CSV is the right cross-reference depends on
which synthetic matrix family (original saturated vs. low-degree) the
weak_partitioned-*.out logs being analyzed came from -- pass
--cyclic-summary explicitly when analyzing the low-degree family:
  python bonus_partitioning_strategies/analyze_weak_partitioned.py \\
      --cyclic-summary ../outputs/csv/weak_scaling_sparse_summary.csv \\
      bonus_partitioning_strategies/outputs/weak_scaling/weak_partitioned-<jobids>.out

Usage:
  python bonus_partitioning_strategies/analyze_weak_partitioned.py [outputs/weak_scaling/weak_partitioned-*.out ...]
  (with no positional args, globs bonus_partitioning_strategies/outputs/weak_scaling/weak_partitioned-*.out;
  default --cyclic-summary is ../outputs/csv/weak_scaling_summary.csv, i.e. the
  original saturated family -- override for the low-degree family)
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import median, mean

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from result_parser import parse_files, chunk_repetitions  # noqa: E402


def load_cyclic_summary(path):
    if not os.path.exists(path):
        return None
    t_by_kp = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["comm"] != "ghost_nccl":
                continue
            t_by_kp[(row["kernel"], int(row["P"]))] = float(row["T_median_ms"])
    return t_by_kp


def summarize(rows, cyclic):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["comm"], r["kernel"], r["P"])].append(r)

    t_by_ck = defaultdict(dict)
    gflops_by_ck = defaultdict(dict)
    agg_gflops_by_ck = defaultdict(dict)
    for (comm, kernel, P), grp in groups.items():
        rep_chunks = chunk_repetitions(grp, P)
        t_reps = [max(r["total_avg_ms"] for r in chunk) for chunk in rep_chunks]
        gflops_reps = [mean(r["effective_gflops"] for r in chunk) for chunk in rep_chunks]
        agg_gflops_reps = [2.0 * sum(r["nnz_local"] for r in chunk) / (t_rep / 1e3) / 1e9
                            for t_rep, chunk in zip(t_reps, rep_chunks)]
        t_by_ck[(comm, kernel)][P] = {"median": median(t_reps), "min": min(t_reps), "max": max(t_reps)}
        gflops_by_ck[(comm, kernel)][P] = median(gflops_reps)
        agg_gflops_by_ck[(comm, kernel)][P] = median(agg_gflops_reps)

    summary_rows = []
    for (comm, kernel), by_p in sorted(t_by_ck.items()):
        if 1 not in by_p:
            continue
        t1 = by_p[1]["median"]
        print(f"\n=== comm={comm} kernel={kernel} ===")
        print(f"{'P':>3} {'T_median_ms':>14} {'T_min_ms':>12} {'T_max_ms':>12} "
              f"{'E_weak':>9} {'mean_rank_gflops':>17} {'aggregate_gflops':>17} "
              f"{'T_cyclic_ghost_nccl_ms':>23} {'speedup_vs_cyclic':>18}")
        for P in sorted(by_p):
            t = by_p[P]
            e_weak = t1 / t["median"] if t["median"] > 0 else float("nan")
            gflops = gflops_by_ck[(comm, kernel)][P]
            agg_gflops = agg_gflops_by_ck[(comm, kernel)][P]
            t_cyclic = cyclic.get((kernel, P)) if cyclic else None
            speedup = (t_cyclic / t["median"]) if (t_cyclic and t["median"] > 0) else None
            print(f"{P:>3} {t['median']:>14.4f} {t['min']:>12.4f} {t['max']:>12.4f} "
                  f"{e_weak:>9.3f} {gflops:>17.4f} {agg_gflops:>17.4f} "
                  f"{(t_cyclic if t_cyclic is not None else float('nan')):>23.4f} "
                  f"{(speedup if speedup is not None else float('nan')):>18.3f}")
            summary_rows.append([comm, kernel, P, t["median"], t["min"], t["max"],
                                  e_weak, gflops, agg_gflops, t_cyclic, speedup])

    return summary_rows


def main():
    argv = sys.argv[1:]
    cyclic_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "csv", "weak_scaling_summary.csv")
    if "--cyclic-summary" in argv:
        i = argv.index("--cyclic-summary")
        cyclic_path = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]

    default_dir = os.path.join(os.path.dirname(__file__), "outputs", "weak_scaling")
    paths = argv or sorted(glob.glob(os.path.join(default_dir, "weak_partitioned-*.out")))
    if not paths:
        print(f"no input files (pass paths, or run from the project root with "
              f"{default_dir}/weak_partitioned-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    cyclic = load_cyclic_summary(cyclic_path)
    if cyclic is None:
        print(f"note: {cyclic_path} not found -- reporting partitioned-only numbers, "
              f"no cyclic cross-reference", file=sys.stderr)
    else:
        print(f"Cross-referencing cyclic comm=ghost_nccl numbers from {cyclic_path}", file=sys.stderr)

    summary_rows = summarize(rows, cyclic)

    csv_dir = os.path.join(os.path.dirname(__file__), "outputs", "csv")
    os.makedirs(csv_dir, exist_ok=True)
    out_csv = os.path.join(csv_dir, "weak_partitioned_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comm", "kernel", "P", "T_median_ms", "T_min_ms", "T_max_ms",
                    "E_weak", "mean_rank_effective_gflops", "aggregate_gflops",
                    "T_cyclic_ghost_nccl_median_ms", "speedup_vs_cyclic_ghost_nccl"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
