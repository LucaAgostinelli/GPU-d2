#!/usr/bin/env python3
"""
shared parser for this project's tagged RESULT lines:
    RESULT comm=<...> kernel=<...> matrix=<...> P=<...> rank=<...> key=value ...
Parses each line into a plain dict keyed by field name.
"""
import re
import sys

_TOKEN_RE = re.compile(r"(\S+)=(\S+)")

_INT_FIELDS = {
    "P", "rank", "nrows_local", "ncols_global", "nnz_local", "n_ghost",
    "M", "N", "nnz_expanded",
}
_FLOAT_FIELDS = {
    "compute_avg_ms", "compute_min_ms", "compute_max_ms", "compute_var_ms", "compute_gflops",
    "eff_bw_gbs", "pct_peak_bw",
    "comm_avg_ms", "comm_min_ms", "comm_max_ms", "comm_var_ms",
    "total_avg_ms", "total_min_ms", "total_max_ms", "total_var_ms",
    "effective_gflops",
    "avg_ms", "min_ms", "max_ms", "gflops",
}


def parse_result_line(line):
    """Parse one 'RESULT key=value ...' line into a dict, or None if the
    line isn't a RESULT line."""
    line = line.strip()
    if not line.startswith("RESULT "):
        return None
    row = {}
    for key, value in _TOKEN_RE.findall(line[len("RESULT "):]):
        if key in _INT_FIELDS:
            row[key] = int(value)
        elif key in _FLOAT_FIELDS:
            row[key] = float(value)
        else:
            row[key] = value
    return row


def parse_files(paths):
    """Parse every RESULT line across the given files, in file order."""
    rows = []
    for path in paths:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                row = parse_result_line(line)
                if row is not None:
                    rows.append(row)
    return rows


def chunk_repetitions(rows_in_order, P):
    """Recover per-rep row groups as consecutive P-sized chunks (file order)."""
    n_full = (len(rows_in_order) // P) * P
    if n_full != len(rows_in_order):
        print(f"warning: {len(rows_in_order)} rows for P={P} is not a "
              f"multiple of P, dropping {len(rows_in_order) - n_full} "
              f"trailing row(s)", file=sys.stderr)
    return [rows_in_order[i:i + P] for i in range(0, n_full, P)]
