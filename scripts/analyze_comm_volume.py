#!/usr/bin/env python3
"""
communication volume per rank, in bytes.

recv_bytes per call: bcast = ncols_global*4; ghost = n_ghost*4;
checkerboard = (ncols_local + nrows_local)*4. Uses the first rep only
(structural, not a timing).

Usage:
  python analyze_comm_volume.py [outputs/strong_scaling/strong-*.out ...]
  (with no args, globs outputs/strong_scaling/strong-*.out)
"""
import csv
import glob
import math
import sys
from collections import defaultdict
from statistics import mean

from result_parser import parse_files, chunk_repetitions

BYTES_PER_ELEMENT = 4  # float32


def factor_grid(P):
    """Mirrors src/proc_grid.cpp's make_proc_grid(): P_r = largest divisor
    of P that is <= sqrt(P)."""
    P_r = int(math.isqrt(P))
    if P_r < 1:
        P_r = 1
    while P_r > 1 and P % P_r != 0:
        P_r -= 1
    return P_r, P // P_r


def local_count(n_global, P, idx):
    """Mirrors src/distribute_matrix.cpp's local_nrows()."""
    return n_global // P + (1 if idx < n_global % P else 0)


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

    groups = defaultdict(list)
    for r in rows:
        groups[(r["comm"], r["matrix"], r["P"])].append(r)

    naive = {}
    ghost = {}
    checkerboard = {}

    for (comm, matrix, P), grp in groups.items():
        chunk = chunk_repetitions(grp, P)[0]

        if comm == "bcast":
            vals = [r["ncols_global"] * BYTES_PER_ELEMENT for r in chunk]
            naive[(matrix, P)] = mean(vals)
        elif comm in ("ghost_mpi", "ghost_nccl"):
            vals = [r["n_ghost"] * BYTES_PER_ELEMENT for r in chunk]
            ghost.setdefault((matrix, P), mean(vals))
        elif comm in ("checkerboard_mpi", "checkerboard_nccl"):
            P_r, P_c = factor_grid(P)
            vals = []
            for r in chunk:
                pc = r["rank"] % P_c
                ncols_local = local_count(r["ncols_global"], P_c, pc)
                vals.append((ncols_local + r["nrows_local"]) * BYTES_PER_ELEMENT)
            checkerboard.setdefault((matrix, P), mean(vals))

    matrices_p = sorted(set(naive) | set(ghost) | set(checkerboard))
    summary_rows = []
    print(f"{'matrix':<20} {'P':>3} {'naive_bytes':>14} {'ghost_bytes':>14} "
          f"{'checkerboard_bytes':>20} {'ghost/naive':>12} {'ckb/naive':>11} {'ckb/ghost':>11}")
    for (matrix, P) in matrices_p:
        n = naive.get((matrix, P))
        g = ghost.get((matrix, P))
        c = checkerboard.get((matrix, P))
        g_n = (g / n) if (g is not None and n) else float("nan")
        c_n = (c / n) if (c is not None and n) else float("nan")
        c_g = (c / g) if (c is not None and g) else float("nan")
        print(f"{matrix:<20} {P:>3} "
              f"{(n if n is not None else float('nan')):>14.0f} "
              f"{(g if g is not None else float('nan')):>14.0f} "
              f"{(c if c is not None else float('nan')):>20.0f} "
              f"{g_n:>12.4f} {c_n:>11.4f} {c_g:>11.4f}")
        summary_rows.append([matrix, P, n, g, c, g_n, c_n, c_g])

    out_csv = "outputs/csv/comm_volume_summary.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "P", "naive_bytes", "ghost_bytes", "checkerboard_bytes",
                    "ghost_over_naive", "checkerboard_over_naive", "checkerboard_over_ghost"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
