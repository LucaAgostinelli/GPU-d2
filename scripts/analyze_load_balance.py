#!/usr/bin/env python3
"""
NNZ-per-rank load-balance stats.

Groups by partitioning family (1d_cyclic, 2d_checkerboard) x matrix x P,
first rep only. Reports min/avg/max/stdev NNZ-per-rank and two imbalance
ratios (max/avg, max/min).

Usage:
  python analyze_load_balance.py [outputs/strong_scaling/strong-*.out outputs/weak_scaling/weak-*.out ...]
  (with no args, globs outputs/strong_scaling/strong-*.out and outputs/weak_scaling/weak-*.out)
"""
import csv
import glob
import sys
from collections import defaultdict
from statistics import mean, stdev

from result_parser import parse_files, chunk_repetitions

FAMILY_OF_COMM = {
    "bcast": "1d_cyclic", "ghost_mpi": "1d_cyclic", "ghost_nccl": "1d_cyclic",
    "checkerboard_mpi": "2d_checkerboard", "checkerboard_nccl": "2d_checkerboard",
}


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        family = FAMILY_OF_COMM.get(r["comm"])
        if family is None:
            continue
        groups[(family, r["matrix"], r["P"])].append(r)

    summary_rows = []
    print(f"{'family':<16} {'matrix':<20} {'P':>3} {'min':>10} {'avg':>10} {'max':>10} "
          f"{'stdev':>10} {'max/avg':>9} {'max/min':>9}")
    for (family, matrix, P), grp in sorted(groups.items()):
        chunk = chunk_repetitions(grp, P)[0]
        nnz_per_rank = sorted(r["nnz_local"] for r in chunk)
        avg = mean(nnz_per_rank)
        mx, mn = max(nnz_per_rank), min(nnz_per_rank)
        sd = stdev(nnz_per_rank) if len(nnz_per_rank) > 1 else 0.0
        max_avg = mx / avg if avg > 0 else float("nan")
        max_min = mx / mn if mn > 0 else float("nan")
        print(f"{family:<16} {matrix:<20} {P:>3} {mn:>10} {avg:>10.1f} {mx:>10} "
              f"{sd:>10.1f} {max_avg:>9.3f} {max_min:>9.3f}")
        summary_rows.append([family, matrix, P, mn, avg, mx, sd, max_avg, max_min])

    return summary_rows


def main():
    paths = sys.argv[1:] or (sorted(glob.glob("outputs/strong_scaling/strong-*.out")) +
                              sorted(glob.glob("outputs/weak_scaling/weak-*.out")))
    if not paths:
        print("no input files (pass paths, or run from the project root with "
              "outputs/strong_scaling/strong-*.out / outputs/weak_scaling/weak-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    summary_rows = summarize(rows)

    out_csv = "outputs/csv/load_balance_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "matrix", "P", "nnz_min", "nnz_avg", "nnz_max", "nnz_stdev",
                    "max_over_avg", "max_over_min"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
