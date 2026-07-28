#!/usr/bin/env python3
"""
summary for the distributed matrix-reading bonus: compares the
baseline rank-0-serial-read+Scatterv path ("serial") against every rank independently reading its own
line-aligned byte range + MPI_Alltoallv redistribution ("chunked").

Parses distributed_matrix_read/outputs/mtx_read-*.out. Each invocation of
bench_mtx_read prints exactly ONE `RESULT_CSV_READ` line (rank 0 only) for
a given (matrix, P).
A (matrix, P) combination that happens to have more than one job's line
(e.g. a resubmission) is reduced by MEDIAN across those lines' fields,
field by field.

Reports, per (matrix, P): serial_mean_ms vs. chunked_mean_ms (the whole
read+distribute pipeline for each strategy), speedup_chunked_over_serial,
and the phase breakdown (serial_read_ms/serial_distribute_ms vs.
chunked_io_ms/chunked_parse_ms/chunked_redistribute_ms) to see where time
actually goes in each strategy. check=OK!/MISMATCH! is carried through
as-is; flagged as MIXED:... if multiple runs of the same (matrix, P)
disagree.

NOTE This script SKIPS any RESULT_CSV_READ line missing `chunked_mean_ms` and reports how
many were skipped.

Usage:
  python analyze_mtx_read.py [path/to/mtx_read-*.out ...]
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict
from statistics import median

_TOKEN_RE = re.compile(r"(\S+)=(\S+)")
_INT_FIELDS = {"P", "reps", "nnz_total_serial", "nnz_total_chunked", "spmv_mismatches"}
_FLOAT_FIELDS = {
    "serial_mean_ms", "serial_min_ms", "chunked_mean_ms", "chunked_min_ms",
    "serial_read_ms", "serial_distribute_ms",
    "chunked_io_ms", "chunked_parse_ms", "chunked_redistribute_ms",
}


def parse_files(paths):
    rows = []
    n_old_schema = 0
    for path in paths:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("RESULT_CSV_READ "):
                    continue
                row = {}
                for key, value in _TOKEN_RE.findall(line[len("RESULT_CSV_READ "):]):
                    if key in _INT_FIELDS:
                        row[key] = int(value)
                    elif key in _FLOAT_FIELDS:
                        row[key] = float(value)
                    else:
                        row[key] = value
                if "chunked_mean_ms" not in row:
                    # older mpiio_*-schema line (superseded implementation,
                    # see module docstring) -- skip, don't silently mix.
                    n_old_schema += 1
                    continue
                rows.append(row)
    if n_old_schema:
        print(f"note: skipped {n_old_schema} older mpiio_*-schema RESULT_CSV_READ "
              f"line(s) (superseded implementation, see module docstring)", file=sys.stderr)
    return rows


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["matrix"], r["P"])].append(r)

    summary_rows = []
    print(f"{'matrix':<20} {'P':>3} {'n':>2} {'serial_ms':>11} {'chunked_ms':>11} "
          f"{'speedup':>9} {'check':>12}")
    for (matrix, P), grp in sorted(groups.items()):
        serial_mean = median(r["serial_mean_ms"] for r in grp)
        chunked_mean = median(r["chunked_mean_ms"] for r in grp)
        serial_read = median(r["serial_read_ms"] for r in grp)
        serial_dist = median(r["serial_distribute_ms"] for r in grp)
        chunked_io = median(r["chunked_io_ms"] for r in grp)
        chunked_parse = median(r["chunked_parse_ms"] for r in grp)
        chunked_redist = median(r["chunked_redistribute_ms"] for r in grp)
        speedup = serial_mean / chunked_mean if chunked_mean > 0 else float("nan")
        checks = sorted(set(r["check"] for r in grp))
        check = checks[0] if len(checks) == 1 else "MIXED:" + ",".join(checks)
        print(f"{matrix:<20} {P:>3} {len(grp):>2} {serial_mean:>11.4f} {chunked_mean:>11.4f} "
              f"{speedup:>9.3f} {check:>12}")
        summary_rows.append([matrix, P, len(grp), serial_mean, chunked_mean, speedup,
                              serial_read, serial_dist, chunked_io, chunked_parse,
                              chunked_redist, check])
    return summary_rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    paths = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "outputs", "mtx_read-*.out")))
    if not paths:
        print("no input files (pass paths, or run with "
              "distributed_matrix_read/outputs/mtx_read-*.out present)", file=sys.stderr)
        sys.exit(1)

    rows = parse_files(paths)
    if not rows:
        print("no RESULT_CSV_READ rows found in given files", file=sys.stderr)
        sys.exit(1)

    summary_rows = summarize(rows)

    out_csv = os.path.join(here, "outputs", "mtx_read_summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["matrix", "P", "n_runs", "serial_mean_ms", "chunked_mean_ms",
                    "speedup_chunked_over_serial",
                    "serial_read_ms", "serial_distribute_ms",
                    "chunked_io_ms", "chunked_parse_ms", "chunked_redistribute_ms",
                    "check"])
        w.writerows(summary_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
