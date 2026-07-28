#!/usr/bin/env python3
"""
weak-scaling synthetic matrix generator.

Generates a square NxN Erdos-Renyi sparse matrix (Matrix Market coordinate
real general), N = rows_per_rank * P, nnz_per_row columns per row drawn
uniformly at random -- constant nnz per rank across P. Pure Python stdlib.

Usage:
  python3 gen_weak_matrix.py --P 1               # single size
  python3 gen_weak_matrix.py --all                # P=1,2,3,4 in one go
  python3 gen_weak_matrix.py --P 2 --rows-per-rank 50000 --nnz-per-row 15
"""
import argparse
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
        f.write(f"%% weak-scaling synthetic matrix: P={P}, "
                f"rows_per_rank={rows_per_rank}, nnz_per_row={nnz_per_row}, seed={seed}\n")
        f.write(f"{n} {n} {nnz_total}\n")
        for row in range(1, n + 1):
            for _ in range(nnz_per_row):
                f.write(f"{row} {randrange(n) + 1} {uniform(-1.0, 1.0):.6f}\n")

    print(f"Wrote {out_path}: {n}x{n}, {nnz_total} nnz "
          f"({nnz_per_row} nnz/row, {nnz_total // P} nnz/rank at P={P})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--P", type=int, help="Rank count this matrix targets (single-size mode)")
    ap.add_argument("--all", action="store_true", help="Generate P=1,2,3,4 in one invocation")
    ap.add_argument("--rows-per-rank", type=int, default=75000)
    ap.add_argument("--nnz-per-row", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, help="Base seed; actual seed is seed+P")
    ap.add_argument("--out", type=str, default=None,
                     help="Output path (single-size mode only; default matrices_synthetic/weak_P<P>.mtx)")
    ap.add_argument("--out-dir", type=str, default="matrices_synthetic",
                     help="Output directory for --all mode")
    args = ap.parse_args()

    if not args.all and args.P is None:
        ap.error("specify --P <n> or --all")

    if args.all:
        for P in (1, 2, 3, 4):
            out_path = os.path.join(args.out_dir, f"weak_P{P}.mtx")
            generate(P, args.rows_per_rank, args.nnz_per_row, args.seed + P, out_path)
    else:
        out_path = args.out or os.path.join(args.out_dir, f"weak_P{args.P}.mtx")
        generate(args.P, args.rows_per_rank, args.nnz_per_row, args.seed + args.P, out_path)


if __name__ == "__main__":
    main()
