# Distributed matrix read (bonus)

Standalone read-phase benchmark comparing two ways of getting from a `.mtx`
file on disk to a distributed `LocalCSR` shard on every rank. It is not
one of the SpMV drivers and is not wired into any of them.

---

## The two methods

| Method | Strategy |
|---|---|
| **serial** | rank 0 reads the whole file (`read_mtx`, `common/matrix.cpp`) and `Scatterv`'s pre-built CSR shards (`src/distribute_matrix.cpp`) — what every other driver in this project uses today |
| **chunked** | every rank independently opens the file, parses the header itself (no `Bcast`), and reads its own line-aligned byte range with a plain `std::ifstream::read` (aligned to line boundaries, no inter-rank handshake needed — the alignment is a pure function of file content) — no MPI collective I/O call anywhere in the read step. Parses the raw buffer directly with `strtol`/`strtof` (no per-line `std::string`/`istringstream`), then redistributes by the same `owner(i) = i % P` rule via `MPI_Alltoallv` — no rank-0 bottleneck, and each rank does its own COO→CSR conversion instead of root doing all P of them |

Each method gets one untimed warmup call before its own timed loop, so
neither carries an unfair OS-page-cache advantage over the other.

---

## Where the code lives

The driver source (`drivers/bench_mtx_read.cpp`) and the chunked-read
implementation (`src/distribute_matrix_chunked.{hpp,cpp}`) live in the
project's shared trees and build as a normal target in the root
`CMakeLists.txt`/`build.sh`.
This directory holds only what's specific to this benchmark: its own SLURM
scripts, analysis script, and (once collected) `outputs/`.

```bash
mpirun -n <P> ./bin/bench_mtx_read <matrix.mtx> [reps]   # reps defaults to 3
```

---

## Result format

Rank 0 prints one `RESULT_CSV_READ` line per invocation: mean/min ms for
each method, a phase breakdown (`serial_read_ms`/`serial_distribute_ms` and
`chunked_io_ms`/`chunked_parse_ms`/`chunked_redistribute_ms`, each the
slowest rank for that phase) to see empirically where time actually goes,
and a real numeric correctness cross-check (`check=OK!`/`MISMATCH!`): a
fixed deterministic x vector is multiplied against each method's `LocalCSR`
shard on every rank, and the resulting y vectors are compared
element-by-element.

---

## Running

All commands below run from the project root.

### Small/medium matrices (`matrices/`, one job, P and REPS given directly)

```bash
sbatch distributed_matrix_read/sbatch/mtx_read_benchmark.sh matrices/<name>.mtx <P> [REPS]
```

### Big matrices (`big_matrices/`, up to 1.02B nnz)

No code changes needed for this scale,
but the small-matrix
script's 5-minute budget is far too tight. One job per matrix, sweeps
P=1..4 internally, `REPS` defaults to 1:

```bash
sbatch distributed_matrix_read/sbatch/mtx_read_benchmark_large.sh big_matrices/<name>.mtx [REPS]

# submit all 10 big matrices as independent jobs
bash distributed_matrix_read/sbatch/submit_big_matrices_mtx_read.sh [REPS]
```

---

## Analysis

```bash
python distributed_matrix_read/analyze_mtx_read.py
```

Reads `distributed_matrix_read/outputs/mtx_read-*.out`, reduces multiple
runs of the same (matrix, P) by median field-by-field, and writes
`distributed_matrix_read/outputs/mtx_read_summary.csv` — one row per
(matrix, P) with both methods' timings, `speedup_chunked_over_serial`, the
phase breakdown, and the carried-through `check` result.
