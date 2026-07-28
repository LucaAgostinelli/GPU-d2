#!/usr/bin/env python3
"""
ACC vs cuSPARSE comparison for the RCM/Fennel/Block prototypes, cross-referenced against the baseline cyclic
driver (comm=ghost_nccl) under both kernels.

Parses both kernel variants for all three prototypes:
  bonus_partitioning_strategies/outputs/rcm/rcm-*.out                (comm=rcm_nccl kernel=acc)
  bonus_partitioning_strategies/outputs/rcm/rcm_cusparse-*.out       (comm=rcm_nccl kernel=cusparse)
  bonus_partitioning_strategies/outputs/fennel/fennel-*.out          (comm=fennel_nccl kernel=acc)
  bonus_partitioning_strategies/outputs/fennel/fennel_cusparse-*.out (comm=fennel_nccl kernel=cusparse)
  bonus_partitioning_strategies/outputs/block/block-*.out            (comm=block_nccl kernel=acc)
  bonus_partitioning_strategies/outputs/block/block_cusparse-*.out   (comm=block_nccl kernel=cusparse)
and ../outputs/csv/strong_scaling_summary.csv for the cyclic reference.

T(P) = max total_avg_ms across ranks, median of reps. Skips a prototype if
its files aren't present yet.

Run from the project root:
  python bonus_partitioning_strategies/analyze_kernel_isolation.py
"""
import csv
import glob
import os
import sys
from collections import defaultdict
from statistics import median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from result_parser import parse_files, chunk_repetitions  # noqa: E402

PROTOTYPES = [
    ("rcm", "rcm-*.out", "rcm_cusparse-*.out"),
    ("fennel", "fennel-*.out", "fennel_cusparse-*.out"),
    ("block", "block-*.out", "block_cusparse-*.out"),
]


def t_by_matrix_p(paths):
    rows = parse_files(paths)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["matrix"], r["P"])].append(r)
    out = {}
    for (matrix, P), grp in groups.items():
        chunks = chunk_repetitions(grp, P)
        t_reps = [max(r["total_avg_ms"] for r in chunk) for chunk in chunks]
        out[(matrix, P)] = median(t_reps)
    return out


def load_strong_scaling(path):
    if not os.path.exists(path):
        return {}
    t = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["comm"], row["kernel"], row["matrix"], int(row["P"]))
            t[key] = float(row["T_median_ms"])
    return t


def fmt(v):
    return f"{v:>10.4f}" if v is not None else f"{'--':>10}"


def main():
    base = os.path.join(os.path.dirname(__file__), "outputs")
    strong = load_strong_scaling(os.path.join(os.path.dirname(__file__), "..",
                                               "outputs", "csv", "strong_scaling_summary.csv"))

    summary_rows = []
    any_found = False

    for name, acc_glob, cus_glob in PROTOTYPES:
        proto_dir = os.path.join(base, name)
        acc = t_by_matrix_p(sorted(glob.glob(os.path.join(proto_dir, acc_glob))))
        cus = t_by_matrix_p(sorted(glob.glob(os.path.join(proto_dir, cus_glob))))
        if not acc and not cus:
            continue
        any_found = True

        matrices = sorted(set(k[0] for k in list(acc) + list(cus)))
        print(f"\n=== {name}: ACC vs cuSPARSE (T(P), ms; lower is better) ===")
        print(f"{'matrix':<16} {'P':>3} {'cyclic_acc':>10} {'cyclic_cus':>10} "
              f"{name + '_acc':>10} {name + '_cus':>10}")
        for matrix in matrices:
            for P in (1, 2, 3, 4):
                cyc_acc = strong.get(("ghost_nccl", "acc", matrix, P))
                cyc_cus = strong.get(("ghost_nccl", "cusparse", matrix, P))
                a, c = acc.get((matrix, P)), cus.get((matrix, P))
                if a is None and c is None:
                    continue
                print(f"{matrix:<16} {P:>3} {fmt(cyc_acc)} {fmt(cyc_cus)} {fmt(a)} {fmt(c)}")
                summary_rows.append([name, matrix, P, cyc_acc, cyc_cus, a, c])

    if not any_found:
        print("no cuSPARSE-or-ACC-variant input files found yet for any prototype "
              f"(looked under {base}/) -- run the sbatch scripts first", file=sys.stderr)
        sys.exit(1)

    csv_dir = os.path.join(base, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    out_csv = os.path.join(csv_dir, "kernel_isolation_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prototype", "matrix", "P", "T_cyclic_acc_ms", "T_cyclic_cusparse_ms",
                     "T_prototype_acc_ms", "T_prototype_cusparse_ms"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
