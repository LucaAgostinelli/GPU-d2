#!/usr/bin/env python3
"""
strong-scaling summary for the big-matrix
sweep, same metrics/formulas as analyze_strong_scaling.py (T(P), speedup,
efficiency, GFLOP/s). Unlike that script, a (comm, kernel, matrix) group
missing P=1 is not dropped -- this dataset is collected opportunistically,
so speedup/efficiency are just left blank for those rows. Also reports
per-rank NNZ imbalance (min/max/avg, max/avg) from the first rep.

Usage:
  python analyze_large_matrices.py [outputs/large_matrices/strong_large_night-*.out ...]
  (with no args, globs outputs/large_matrices/strong_large_night-*.out and
  outputs/large_matrices/strong_large-*.out)
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
    imbalance_by_ckm = defaultdict(dict)
    for (comm, kernel, matrix, P), grp in groups.items():
        rep_chunks = chunk_repetitions(grp, P)
        t_reps = [max(r["total_avg_ms"] for r in chunk) for chunk in rep_chunks]
        gflops_reps = [2.0 * sum(r["nnz_local"] for r in chunk) / (t_rep / 1e3) / 1e9
                        for t_rep, chunk in zip(t_reps, rep_chunks)]
        t_by_ckm[(comm, kernel, matrix)][P] = {
            "median": median(t_reps), "min": min(t_reps), "max": max(t_reps),
        }
        gflops_by_ckm[(comm, kernel, matrix)][P] = median(gflops_reps)

        nnz_per_rank = sorted(r["nnz_local"] for r in rep_chunks[0])
        nnz_avg = sum(nnz_per_rank) / len(nnz_per_rank)
        imbalance_by_ckm[(comm, kernel, matrix)][P] = {
            "nnz_min": min(nnz_per_rank), "nnz_max": max(nnz_per_rank),
            "nnz_avg": nnz_avg,
            "imbalance": (max(nnz_per_rank) / nnz_avg) if nnz_avg > 0 else float("nan"),
        }

    summary_rows = []
    for (comm, kernel, matrix), by_p in sorted(t_by_ckm.items()):
        t1 = by_p[1]["median"] if 1 in by_p else None
        print(f"\n=== {matrix}  comm={comm} kernel={kernel} ===")
        print(f"{'P':>3} {'T_median_ms':>14} {'T_min_ms':>12} {'T_max_ms':>12} "
              f"{'speedup':>9} {'efficiency':>11} {'gflops':>10} {'imbalance':>10}")
        for P in sorted(by_p):
            t = by_p[P]
            speedup = (t1 / t["median"]) if (t1 and t["median"] > 0) else None
            efficiency = (speedup / P) if speedup is not None else None
            gflops = gflops_by_ckm[(comm, kernel, matrix)][P]
            imb = imbalance_by_ckm[(comm, kernel, matrix)][P]
            print(f"{P:>3} {t['median']:>14.4f} {t['min']:>12.4f} {t['max']:>12.4f} "
                  f"{(speedup if speedup is not None else float('nan')):>9.3f} "
                  f"{(efficiency if efficiency is not None else float('nan')):>11.3f} "
                  f"{gflops:>10.2f} {imb['imbalance']:>10.4f}")
            summary_rows.append([matrix, comm, kernel, P, t["median"], t["min"], t["max"],
                                  speedup, efficiency, gflops,
                                  imb["nnz_min"], imb["nnz_max"], imb["nnz_avg"], imb["imbalance"]])

    return summary_rows


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob("outputs/large_matrices/strong_large_night-*.out") +
        glob.glob("outputs/large_matrices/strong_large-*.out")
    )
    if not paths:
        print("no input files (pass paths, or run from the project root with "
              "outputs/large_matrices/strong_large_night-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    summary_rows = summarize(rows)

    out_csv = "outputs/csv/large_matrices_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "comm", "kernel", "P", "T_median_ms", "T_min_ms", "T_max_ms",
                    "speedup", "efficiency", "aggregate_effective_gflops",
                    "nnz_min", "nnz_max", "nnz_avg", "imbalance"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
