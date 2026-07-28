#!/usr/bin/env python3
"""
strong-scaling summary: speedup, efficiency,
GFLOP/s per (comm, kernel, matrix, P).

T(P) = max over ranks of total_avg_ms, median across REPS. Speedup
S(P) = T(1)/T(P), efficiency E(P) = S(P)/P, both within the same
(comm, kernel) family. GFLOP/s(P) = 2*(sum of nnz_local across ranks)/T(P)
-- total work over the wall-clock bottleneck, not a sum of each rank's own
gflops (which would overstate throughput under load imbalance).

Usage:
  python analyze_strong_scaling.py [outputs/strong_scaling/strong-*.out ...]
  (with no args, globs outputs/strong_scaling/strong-*.out)
"""
import csv
import glob
import sys
from collections import defaultdict
from statistics import median

from result_parser import parse_files, chunk_repetitions


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["comm"], r["kernel"], r["matrix"], r["P"])].append(r)

    t_by_ckm = defaultdict(dict)
    gflops_by_ckm = defaultdict(dict)
    for (comm, kernel, matrix, P), grp in groups.items():
        rep_chunks = chunk_repetitions(grp, P)
        t_reps = [max(r["total_avg_ms"] for r in chunk) for chunk in rep_chunks]
        gflops_reps = [2.0 * sum(r["nnz_local"] for r in chunk) / (t_rep / 1e3) / 1e9
                        for t_rep, chunk in zip(t_reps, rep_chunks)]
        t_by_ckm[(comm, kernel, matrix)][P] = {
            "median": median(t_reps), "min": min(t_reps), "max": max(t_reps),
        }
        gflops_by_ckm[(comm, kernel, matrix)][P] = median(gflops_reps)

    summary_rows = []
    for (comm, kernel, matrix), by_p in sorted(t_by_ckm.items()):
        if 1 not in by_p:
            continue
        t1 = by_p[1]["median"]
        print(f"\n=== {matrix}  comm={comm} kernel={kernel} ===")
        print(f"{'P':>3} {'T_median_ms':>14} {'T_min_ms':>12} {'T_max_ms':>12} "
              f"{'speedup':>9} {'efficiency':>11} {'gflops':>10}")
        for P in sorted(by_p):
            t = by_p[P]
            speedup = t1 / t["median"] if t["median"] > 0 else float("nan")
            efficiency = speedup / P
            gflops = gflops_by_ckm[(comm, kernel, matrix)][P]
            print(f"{P:>3} {t['median']:>14.4f} {t['min']:>12.4f} {t['max']:>12.4f} "
                  f"{speedup:>9.3f} {efficiency:>11.3f} {gflops:>10.2f}")
            summary_rows.append([matrix, comm, kernel, P, t["median"], t["min"], t["max"],
                                  speedup, efficiency, gflops])

    return summary_rows


def main():
    paths = sys.argv[1:] or sorted(glob.glob("outputs/strong_scaling/strong-*.out"))
    if not paths:
        print("no input files (pass paths, or run from the project root with "
              "outputs/strong_scaling/strong-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    summary_rows = summarize(rows)

    out_csv = "outputs/csv/strong_scaling_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "comm", "kernel", "P", "T_median_ms", "T_min_ms", "T_max_ms",
                    "speedup", "efficiency", "aggregate_effective_gflops"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
