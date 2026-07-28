#!/usr/bin/env python3
"""
low-degree weak-scaling synthetic matrix generator.

Companion to gen_weak_matrix.py: same constant-nnz-per-rank scheme, but
nnz_per_row=2 (vs. 20) and rows_per_rank=750000 (vs. 75000), so the expected
ghost fraction frac(P) = 1 - exp(-nnz_per_row/P) actually varies across
P=1..4 instead of saturating instantly. Pure Python stdlib (no numpy: see
gen_weak_matrix.py).

Usage:
  python3 gen_weak_matrix_sparse.py --P 1               # single size
  python3 gen_weak_matrix_sparse.py --all                # P=1,2,3,4 in one go
  python3 gen_weak_matrix_sparse.py --P 2 --rows-per-rank 500000 --nnz-per-row 3
"""
import argparse
import math
import os
import random


def generate(P, rows_per_rank, nnz_per_row, seed, out_path):
    n = rows_per_rank * P
    nnz_total = n * nnz_per_row
    rng = random.Random(seed)
    randrange = rng.randrange
    uniform = rng.uniform

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write(f"%% weak-scaling synthetic matrix (low-degree variant): P={P}, "
                f"rows_per_rank={rows_per_rank}, nnz_per_row={nnz_per_row}, seed={seed}\n")
        f.write(f"{n} {n} {nnz_total}\n")
        for row in range(1, n + 1):
            for _ in range(nnz_per_row):
                f.write(f"{row} {randrange(n) + 1} {uniform(-1.0, 1.0):.6f}\n")

    frac_note = ""
    if P > 1:
        frac = 1.0 - math.exp(-nnz_per_row / P)
        frac_note = f", expected ghost fraction ~= {frac:.1%}"
    print(f"Wrote {out_path}: {n}x{n}, {nnz_total} nnz "
          f"({nnz_per_row} nnz/row, {nnz_total // P} nnz/rank at P={P}){frac_note}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=int, help="Rank count this matrix targets (single-size mode)")
    ap.add_argument("--all", action="store_true", help="Generate P=1,2,3,4 in one invocation")
    ap.add_argument("--rows-per-rank", type=int, default=750000)
    ap.add_argument("--nnz-per-row", type=int, default=2)
    ap.add_argument("--seed", type=int, default=142, help="Base seed; actual seed is seed+P")
    ap.add_argument("--out", type=str, default=None,
                     help="Output path (single-size mode only; default "
                          "matrices_synthetic/weak_sparse_P<P>.mtx)")
    ap.add_argument("--out-dir", type=str, default="matrices_synthetic",
                     help="Output directory for --all mode")
    args = ap.parse_args()

    if not args.all and args.P is None:
        ap.error("specify --P <n> or --all")

    if args.all:
        for P in (1, 2, 3, 4):
            out_path = os.path.join(args.out_dir, f"weak_sparse_P{P}.mtx")
            generate(P, args.rows_per_rank, args.nnz_per_row, args.seed + P, out_path)
    else:
        out_path = args.out or os.path.join(args.out_dir, f"weak_sparse_P{args.P}.mtx")
        generate(args.P, args.rows_per_rank, args.nnz_per_row, args.seed + args.P, out_path)


if __name__ == "__main__":
    main()
