#!/usr/bin/env python3
"""
serial CPU baseline summary, cross-referenced against
strong-scaling speedups at a few representative configurations. Pulls T(P)
from BOTH outputs/csv/strong_scaling_summary.csv (the 10 small matrices)
and outputs/csv/large_matrices_summary.csv (the big matrices). 
The two matrix sets are disjoint, so both are
simply merged into one (comm, kernel, matrix, P) -> T_median_ms lookup and
every baseline row (whichever set it belongs to) gets a speedup column if a
matching GPU run exists in either file.

Usage:
  python analyze_baseline.py [outputs/baseline/baseline-*.out ...]
  (with no args, globs outputs/baseline/baseline-*.out)
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import median

from result_parser import parse_files

SPEEDUP_CONFIGS = [
    ("bcast", "cusparse", 1, "speedup_bcast_cusparse_P1"),
    ("ghost_mpi", "cusparse", 1, "speedup_ghost_cusparse_P1"),
    ("ghost_mpi", "cusparse", 4, "speedup_ghost_cusparse_P4"),
    ("ghost_nccl", "cusparse", 4, "speedup_ghost_nccl_cusparse_P4"),
]

SCALING_CSVS = [
    "outputs/csv/strong_scaling_summary.csv",
    "outputs/csv/large_matrices_summary.csv",
]


def load_scaling_times(path):
    if not os.path.exists(path):
        return None
    t_by_ckmp = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["comm"], row["kernel"], row["matrix"], int(row["P"]))
            t_by_ckmp[key] = float(row["T_median_ms"])
    return t_by_ckmp


def load_all_scaling_times(paths):
    """Merge every CSV in `paths` into one lookup. The matrix sets in
    strong_scaling_summary.csv and large_matrices_summary.csv are disjoint,
    so key collisions aren't expected in practice; later paths win if any
    ever occur."""
    combined = {}
    any_found = False
    for path in paths:
        t = load_scaling_times(path)
        if t is None:
            print(f"note: {path} not found -- skipping that cross-reference",
                  file=sys.stderr)
            continue
        any_found = True
        combined.update(t)
    return combined if any_found else None


def main():
    paths = sys.argv[1:] or sorted(glob.glob("outputs/baseline/baseline-*.out"))
    if not paths:
        print("no input files (pass paths, or run from the project root with "
              "outputs/baseline/baseline-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    by_matrix = defaultdict(list)
    for r in rows:
        by_matrix[r["matrix"]].append(r["avg_ms"])

    scaling = load_all_scaling_times(SCALING_CSVS)
    if scaling is None:
        print("note: no scaling summary CSVs found -- reporting "
              "baseline-only numbers, no speedup cross-reference", file=sys.stderr)

    summary_rows = []
    print(f"{'matrix':<20} {'avg_ms':>10} " +
          " ".join(f"{label:>30}" for _, _, _, label in SPEEDUP_CONFIGS))
    for matrix, avg_list in sorted(by_matrix.items()):
        avg_ms = median(avg_list)
        speedups = []
        for comm, kernel, P, _ in SPEEDUP_CONFIGS:
            t = scaling.get((comm, kernel, matrix, P)) if scaling else None
            speedups.append(avg_ms / t if t else None)
        print(f"{matrix:<20} {avg_ms:>10.4f} " +
              " ".join(f"{(s if s is not None else float('nan')):>30.3f}" for s in speedups))
        summary_rows.append([matrix, avg_ms] + speedups)

    out_csv = "outputs/csv/baseline_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "cpu_avg_ms"] + [label for _, _, _, label in SPEEDUP_CONFIGS])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
