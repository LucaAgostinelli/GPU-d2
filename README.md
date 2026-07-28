# Distributed SpMV — Multi-GPU MPI Investigation (4x NVIDIA A30)

Experimental study of distributed-memory **Sparse Matrix-Vector
Multiplication (SpMV)** across multiple GPUs, using GPU-aware MPI and
NCCL. Covers 1D cyclic and 2D checkerboard partitioning, selective
ghost-entry exchange, and several communication transports, evaluated on
4x NVIDIA A30 GPUs (single node) across two SuiteSparse matrix sets, a
small one and a large one (see [Two matrix sets](#two-matrix-sets)).

---

## Quick navigation

| What you want | Where to look |
|---|---|
| **Benchmark numbers** (time, speedup, efficiency, GFLOP/s) | `outputs/csv/*.csv` |
| **Raw benchmark output** (one file per job) | `outputs/<category>/*.out` (`.err` for stderr) |
| **Generated plots** | `outputs/figures/*.png` |
| **Alternative partitioning strategies** (RCM, Fennel/LDG, Block) | `bonus_partitioning_strategies/` — own README |
| **Distributed matrix-read benchmark** | `distributed_matrix_read/` — own README |
| **Compute-node hardware/software profile** | `node_architecture/` — own README |

---

## Two matrix sets

This project benchmarks two separate matrix sets, kept in two separate
directories:

- **`matrices/`** — 10 small matrices, the same set used in the
  single-GPU predecessor project (see [Dataset](#dataset)).
- **`big_matrices/`** — 10 additional, much larger matrices (up to
  1.02B nnz), used for a dedicated large-scale strong-scaling
  investigation.

Both sets were fully benchmarked, and both result sets are still shipped
under `outputs/`. **The written report, however, only presents the
big-matrix numbers.** At that scale, communication genuinely dominates
and the strong/weak-scaling trends are far more informative — several
small matrices finish so fast that fixed per-call overhead swamps the
actual communication pattern being studied, which makes it harder to see
the effect the report is about. The small-matrix data and driver
infrastructure were kept anyway because they're a much faster
correctness/smoke-test path (minutes, vs. up to 2-hour jobs and multi-GB
downloads for the big set).

---

## Repository layout

```
.
├── common/                 # matrix I/O (Matrix Market -> COO -> CSR), CPU reference SpMV,
│                           # random vector generation, the mmio library (CPU baseline only)
│   └── include/            # headers for the above
├── kernels/                # local SpMV kernels
│   ├── cusparse.cu/.hpp    # cuSPARSE
│   └── acc.cu/.hpp         # adaptive dispatcher, LINE-enhance vs. FLAT (Chu et al., 2024)
├── src/                    # distributed-SpMV logic
│   ├── distribute_matrix.cpp          # 1D cyclic partitioning: owner(i) = i mod P
│   ├── ghost_exchange.cpp             # 1D selective ghost-entry exchange
│   ├── distribute_matrix_2d.cpp       # 2D checkerboard partitioning
│   ├── proc_grid.cpp                  # P_r x P_c process grid for 2D
│   ├── distribute_matrix_chunked.cpp  # parallel (non rank-0-serial) matrix read
│   ├── {rcm,fennel,block}_partition.cpp        # alternative row/column ownership rules (bonus)
│   ├── distribute_matrix_partitioned.cpp       # generalizes distribute_matrix.cpp to any of the above
│   ├── ghost_exchange_mapped.cpp               # generalizes ghost_exchange.cpp to any of the above
│   └── include/                       # headers, plus gpu_binding.hpp, nccl_bootstrap.hpp,
│                                       # joint_benchmark.hpp (shared warmup+benchmark harness)
├── drivers/                # one thin main() per driver
│   ├── spmv_bcast(.cu/_acc.cu)        # full-broadcast reference
│   ├── spmv_cpu_baseline.cpp          # serial single-core CPU reference
│   ├── bench_mtx_read.cpp             # read-phase benchmark (see distributed_matrix_read/)
│   ├── 1d/                            # 1D ghost exchange — MPI and NCCL, cuSPARSE and ACC
│   ├── 2d/                            # 2D checkerboard — MPI and NCCL, cuSPARSE and ACC
│   └── partitioning/                  # RCM / Fennel / Block variants (bonus) — NCCL, cuSPARSE and ACC
├── scripts/                # Python analysis + weak-scaling matrix generators
├── sbatch/                 # SLURM job scripts
├── outputs/                # results: outputs/<category>/*.out|.err, outputs/csv/*.csv,
│                           # outputs/figures/*.png
├── bonus_partitioning_strategies/     # RCM/Fennel/Block investigation — own README
├── distributed_matrix_read/           # parallel matrix-read benchmark — own README
├── node_architecture/                 # compute-node hardware profile — own README
├── CMakeLists.txt
└── build.sh
```

---

## Build

```bash
bash build.sh          # cmake + make
bash build.sh clean    # wipe build/ and bin/, rebuild from scratch
```

Requires `CUDA/12.3.2`, `OpenMpi/4.1.5-CUDA-12.3.2`, and
`CMake/3.26.3-GCCcore-12.3.0` (loaded automatically by `build.sh`),
targeting `sm_80` (NVIDIA A30, Ampere). NCCL is located automatically
inside the cluster's NVHPC module tree; if it isn't found, the NCCL-based
targets are skipped with a warning instead of failing the whole build.
`spmv_cpu_baseline` needs no CUDA/MPI at all. Binaries land in `bin/`.

---

## Run

Matrix data isn't shipped (too large) — place the small matrix set under
`matrices/` and the big matrix set under `big_matrices/` (see
[Dataset](#dataset) for exactly which files each expects). Weak scaling
needs synthetic matrices instead, generated once, pure Python stdlib,
independent of the small/big split above:

```bash
python scripts/gen_weak_matrix.py --all
python scripts/gen_weak_matrix_sparse.py --all
```

### Small matrices (`matrices/`, the 10 matrices used in Deliverable 1)

Cheap enough that a full 9-driver sweep at one (matrix, P) fits comfortably
inside a short SLURM job:

```bash
# strong scaling: one job per (matrix, P), all 9 GPU/MPI drivers, P in {1,2,3,4}
sbatch sbatch/strong_scaling.sh matrices/<name>.mtx <P> [REPS]

# submit the full grid at once: every small matrix x every P (40 jobs)
bash sbatch/submit_all_matrices.sh sbatch/strong_scaling.sh [REPS]

# serial CPU reference, one matrix at a time, no P needed
sbatch sbatch/cpu_baseline.sh matrices/<name>.mtx [reps]
bash sbatch/submit_all_matrices.sh sbatch/cpu_baseline.sh [reps]   # all 10 matrices
```

### Big matrices (`big_matrices/`, up to 1.02B nnz — what the report's results come from)

A full multi-driver sweep per job isn't
practical here — these scripts run **one driver at a time**, sweeping P
internally instead of one job per (matrix, P):

```bash
# one matrix, one driver, single P -- quick calibration/check
sbatch sbatch/strong_scaling_large.sh big_matrices/<name>.mtx <P> [REPS]

# one matrix, one driver, full P=1..4 sweep, unattended (up to 2h)
sbatch sbatch/strong_scaling_large_night.sh big_matrices/<name>.mtx [REPS]

# submit every big matrix as an independent job, for one driver
bash sbatch/submit_big_matrices_night.sh [REPS] [DRIVER]

# serial CPU reference on the big matrices (overrides cpu_baseline.sh's own partition/time budget)
bash sbatch/submit_big_matrices_baseline.sh [REPS]
```

### Weak scaling (uses the synthetic matrices generated above, independent of the small/big split)

```bash
sbatch sbatch/weak_scaling.sh <P> [REPS]
```

### Running a single driver directly (either matrix set)

```bash
mpirun -n <P> ./bin/<driver> <matrix.mtx>
```

`strong_scaling.sh`/`weak_scaling.sh` skip the 2D drivers at P=3 (no
non-degenerate square-ish process-grid factorization).

---

## Reproduce CSV results / plots

Each script parses `RESULT ...` lines out of the corresponding
`outputs/<category>/*.out` logs and writes one headered CSV into
`outputs/csv/`. All run from the project root with no arguments (they glob
`outputs/<category>/` by default) unless noted:

| Script | Writes |
|---|---|
| `analyze_strong_scaling.py` | `strong_scaling_summary.csv` (speedup, efficiency, GFLOP/s per comm/kernel/matrix/P) |
| `analyze_weak_scaling.py` | `weak_scaling_summary.csv` (weak-scaling efficiency per comm/kernel/P) — pass explicit file paths, not the default glob, when the two synthetic matrix families are both present (see `scripts/gen_weak_matrix_sparse.py`) |
| `analyze_baseline.py` | `baseline_summary.csv` (speedup over the serial CPU reference) |
| `analyze_load_balance.py` | `load_balance_summary.csv` (NNZ-per-rank min/avg/max/imbalance) |
| `analyze_comm_volume.py` | `comm_volume_summary.csv` (bytes moved per rank, naive vs. 1D-ghost vs. 2D) |
| `analyze_large_matrices.py` | `large_matrices_summary.csv` (same metrics as strong scaling, big-matrix set) |
| `plot_charts.py` | `outputs/figures/*.png`, from the CSVs above |

```bash
python scripts/analyze_strong_scaling.py
python scripts/plot_charts.py
```

---

## Drivers at a glance

The table below covers the 11 core drivers — every communication-strategy
x kernel-choice combination, plus the CPU baseline — named systematically
(`spmv_<comm>_<kernel>`):

| Binary | Communication | Kernel |
|---|---|---|
| `spmv_bcast` | full `MPI_Bcast` every call (reference) | cuSPARSE |
| `spmv_bcast_acc` | full `MPI_Bcast` every call | ACC |
| `spmv_ghost_cusparse` | 1D ghost exchange, MPI point-to-point | cuSPARSE |
| `spmv_ghost_acc` | 1D ghost exchange, MPI point-to-point | ACC |
| `spmv_ghost_nccl_cusparse` | 1D ghost exchange, NCCL | cuSPARSE |
| `spmv_ghost_nccl_acc` | 1D ghost exchange, NCCL | ACC |
| `spmv_2d_cusparse` | 2D checkerboard, MPI Bcast/Reduce | cuSPARSE |
| `spmv_2d_acc` | 2D checkerboard, MPI Bcast/Reduce | ACC |
| `spmv_2d_nccl_cusparse` | 2D checkerboard, NCCL | cuSPARSE |
| `spmv_2d_nccl_acc` | 2D checkerboard, NCCL | ACC |
| `spmv_cpu_baseline` | none (serial CPU reference) | — |

Every GPU/MPI driver takes a matrix path as its only argument
(`mpirun -n <P> ./bin/<driver> <matrix.mtx>`), runs a joint
warmup+benchmark loop (10 warmup + 30 timed reps of exchange-then-kernel),
prints one `RESULT` line per rank, then validates its result against a CPU
reference before exiting.

**On top of these 11, there are 6 additional bonus drivers**
(`spmv_{rcm,fennel,block}_nccl_{acc,cusparse}`) implementing three
alternative partitioning strategies instead of the baseline cyclic rule —
see `bonus_partitioning_strategies/`'s own README for their table and
usage.

---

## Dataset

**Only the big-matrix numbers below are used in the written report** (see
[Two matrix sets](#two-matrix-sets) for why) — the small-matrix set is
still fully benchmarked and shipped under `outputs/`, but it's not what
the report's figures/tables are drawn from.

### Big matrices (`big_matrices/`, up to 1.02B nnz)

10 large matrices  from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) used for the large-matrix strong-scaling
investigation, well beyond the small set's scale:

| Matrix | Rows | NNZ | avg nnz/row |
|---|---:|---:|---:|
| Queen_4147 | 4,147,110 | 329,499,284 | 79.45 |
| arabic-2005 | 22,744,080 | 639,999,458 | 28.14 |
| com-Orkut | 3,072,441 | 234,370,166 | 76.28 |
| europe_osm | 50,912,018 | 108,109,320 | 2.12 |
| kmer_V1r | 214,005,017 | 465,410,904 | 2.17 |
| mawi_201512020330 | 226,196,185 | 480,047,894 | 2.12 |
| mycielskian19 | 393,215 | 903,194,710 | 2296.95 |
| nlpkkt240 | 27,993,600 | 774,472,352 | 27.67 |
| uk-2005 | 39,459,925 | 936,364,282 | 23.73 |
| webbase-2001 | 118,142,155 | 1,019,903,190 | 8.63 |

### Small matrices (`matrices/`, used in Deliverable 1)

10 matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/),
the same set used in the single-GPU predecessor project — still fully
benchmarked here (see [Two matrix sets](#two-matrix-sets)), but not what
the report's numbers come from:

| Matrix | Rows | NNZ | avg nnz/row |
|---|---:|---:|---:|
| amazon0302 | 262,111 | 823,251 | 3.14 |
| ASIC_680k | 682,862 | 3,871,773 | 5.67 |
| cage13 | 445,315 | 7,479,343 | 16.80 |
| cit-Patents | 3,774,768 | 11,012,632 | 2.92 |
| crankseg_2 | 63,838 | 14,148,858 |  221.64 |
| parabolic_fem | 525,825 | 3,674,625 |  6.99 |
| poisson3Da | 13,514 | 352,762 | 26.10 |
| roadNet-CA | 1,971,281 | 3,688,810 |  1.87 |
| thermal2 | 1,228,045 | 8,580,313 |  6.99 |
| webbase-1M | 1,000,005 | 3,105,536 | 3.11 |

---

## Additional investigations

Each of these lives in its own directory with its own README, SLURM
scripts, and (once collected) `outputs/`:

- **`bonus_partitioning_strategies/`** — three alternative row/column
  partitioning strategies (RCM, Fennel/LDG, classical 1D-Block) compared
  against the baseline cyclic rule.
- **`distributed_matrix_read/`** — compares rank-0-serial-read+scatter
  against a fully parallel, per-rank chunked read for loading a `.mtx`
  file.
- **`node_architecture/`** — one-shot hardware/software profile of the
  compute node these benchmarks run on.
