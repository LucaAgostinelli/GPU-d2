#!/usr/bin/env python3
"""
weak-scaling summary, per (comm, kernel, P).

Groups by (comm, kernel, P) -- each P has its own synthetic matrix, so no
shared matrix to group by. E_weak(P) = T(1)/T(P) (not divided by P again).
Two GFLOP/s columns: mean_rank_effective_gflops (mean of each rank's own
2*nnz_local_i/T_i -- per-GPU throughput) and aggregate_gflops
(2*(sum of nnz_local)/T_max -- total work over the wall-clock bottleneck).

Usage:
  python analyze_weak_scaling.py [outputs/weak_scaling/weak-*.out ...]
  (with no args, globs outputs/weak_scaling/weak-*.out)
"""
import csv
import glob
import sys
from collections import defaultdict
from statistics import median, mean

from result_parser import parse_files, chunk_repetitions


def summarize(rows):
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
              f"{'E_weak':>9} {'mean_rank_gflops':>17} {'aggregate_gflops':>17}")
        for P in sorted(by_p):
            t = by_p[P]
            e_weak = t1 / t["median"] if t["median"] > 0 else float("nan")
            gflops = gflops_by_ck[(comm, kernel)][P]
            agg_gflops = agg_gflops_by_ck[(comm, kernel)][P]
            print(f"{P:>3} {t['median']:>14.4f} {t['min']:>12.4f} {t['max']:>12.4f} "
                  f"{e_weak:>9.3f} {gflops:>17.4f} {agg_gflops:>17.4f}")
            summary_rows.append([comm, kernel, P, t["median"], t["min"], t["max"],
                                  e_weak, gflops, agg_gflops])

    return summary_rows


def main():
    paths = sys.argv[1:] or sorted(glob.glob("outputs/weak_scaling/weak-*.out"))
    if not paths:
        print("no input files (pass paths, or run from the project root with "
              "outputs/weak_scaling/weak-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    summary_rows = summarize(rows)

    out_csv = "outputs/csv/weak_scaling_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comm", "kernel", "P", "T_median_ms", "T_min_ms", "T_max_ms",
                    "E_weak", "mean_rank_effective_gflops", "aggregate_gflops"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
