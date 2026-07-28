#!/usr/bin/env python3
"""
Parses bonus_partitioning_strategies/outputs/rcm/rcm-*.out (comm=rcm_nccl kernel=acc
RESULT lines, from sbatch/rcm_small_matrices.sh) and cross-references
../outputs/csv/strong_scaling_summary.csv to
report a direct, one-variable-changed comparison against comm=ghost_nccl
kernel=acc at the SAME matrix/P -- same NCCL ghost-exchange transport and
ACC kernel, only the partitioning (RCM+NNZ-balanced blocks vs. baseline
cyclic i mod P) differs. Also reports n_ghost as a fraction of nrows_local
(ghost saturation), directly comparable to the cyclic driver's own n_ghost
in strong_scaling_summary.csv.

Run from the root directory:
  python bonus_partitioning_strategies/analyze_rcm.py
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from result_parser import parse_files, chunk_repetitions  # noqa: E402


def load_strong_scaling(path):
    if not os.path.exists(path):
        return None
    t_by_ckmp = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["comm"], row["kernel"], row["matrix"], int(row["P"]))
            t_by_ckmp[key] = float(row["T_median_ms"])
    return t_by_ckmp


def main():
    rcm_dir = os.path.join(os.path.dirname(__file__), "outputs", "rcm")
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(rcm_dir, "rcm-*.out")))
    if not paths:
        print(f"no input files (pass paths, or run from the project root with "
              f"{rcm_dir}/rcm-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT rows found in given files", file=sys.stderr)
        sys.exit(1)

    strong_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "csv",
                                "strong_scaling_summary.csv")
    strong = load_strong_scaling(strong_path)
    if strong is None:
        print(f"note: {strong_path} not found -- reporting RCM-only numbers, "
              f"no cyclic cross-reference", file=sys.stderr)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["matrix"], r["P"])].append(r)

    summary_rows = []
    print(f"{'matrix':<18} {'P':>3} {'T_rcm_ms':>12} {'T_cyclic_ms':>14} "
          f"{'speedup_vs_cyclic':>18} {'ghost_frac_rcm':>15}")
    for (matrix, P), grp in sorted(groups.items()):
        rep_chunks = chunk_repetitions(grp, P)
        t_reps = [max(r["total_avg_ms"] for r in chunk) for chunk in rep_chunks]
        t_rcm = median(t_reps)

        # Ghost fraction: n_ghost / nrows_local, averaged over ranks within
        # one representative rep (both are static per-rank quantities, not
        # timing samples -- no need to median across reps).
        ghost_fracs = [r["n_ghost"] / r["nrows_local"] for r in rep_chunks[0] if r["nrows_local"] > 0]
        ghost_frac = sum(ghost_fracs) / len(ghost_fracs) if ghost_fracs else float("nan")

        t_cyclic = strong.get(("ghost_nccl", "acc", matrix, P)) if strong else None
        speedup = (t_cyclic / t_rcm) if (t_cyclic and t_rcm > 0) else None

        print(f"{matrix:<18} {P:>3} {t_rcm:>12.4f} "
              f"{(t_cyclic if t_cyclic else float('nan')):>14.4f} "
              f"{(speedup if speedup is not None else float('nan')):>18.3f} "
              f"{ghost_frac:>15.4f}")
        summary_rows.append([matrix, P, t_rcm, t_cyclic, speedup, ghost_frac])

    csv_dir = os.path.join(os.path.dirname(__file__), "outputs", "csv")
    out_csv = os.path.join(csv_dir, "rcm_vs_cyclic_summary.csv")
    os.makedirs(csv_dir, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "P", "T_rcm_median_ms", "T_cyclic_ghost_nccl_acc_median_ms",
                     "speedup_rcm_over_cyclic", "ghost_fraction_rcm"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
