# Intelligent partitioning strategies (bonus)

Three additional, opt-in row/column partitioning strategies, alternative to
the baseline 1D cyclic rule `owner(i) = i mod P`. Cyclic partitioning is
data-agnostic (it ignores the matrix's sparsity structure entirely) so on
matrices with exploitable locality it scatters neighboring rows across
every rank, maximizing ghost-exchange volume instead of minimizing it. Each
strategy here computes a different, still-square-matrix row/column
ownership up front, then reuses the same ghost-exchange/NCCL machinery as
the baseline 1D driver, isolating partitioning quality as the only variable.

---

## The three strategies

- **RCM**: Reverse Cuthill-McKee reordering of the matrix's sparsity-pattern
  graph, then P contiguous, NNZ-balanced blocks over that reordering.
  Clusters mutually-adjacent rows together before slicing.
- **Fennel/LDG**: streaming greedy edge-cut partitioning: each vertex is
  assigned directly to whichever partition already holds the most of its
  neighbors, weighted by a load-balance penalty. No reordering step.
- **Block**: classical 1D-Block: contiguous row ranges over the matrix's
  *original* row order, split as evenly as possible by row count. No NNZ
  balancing, no reordering, no view of the sparsity pattern at all.

---

## Where the code lives

This directory holds only the material specific to these three strategies
that isn't a normal CMake build target: analysis scripts, SLURM scripts,
and (once collected) their own `outputs/`. The actual implementation lives
in the project's shared trees and builds as part of the root
`CMakeLists.txt`/`build.sh`, same as every other driver:

| File | Role |
|---|---|
| `src/{rcm,fennel,block}_partition.{hpp,cpp}` | partition computation (root-only, broadcasts `owner_of[]`/`local_of[]`/`block_start[]` to every rank) |
| `src/distribute_matrix_partitioned.{hpp,cpp}` | generalizes `distribute_matrix.hpp` to an arbitrary `owner_of[]`/`local_of[]` instead of the hardcoded `i mod P` |
| `src/ghost_exchange_mapped.{hpp,cpp}` | same generalization for `ghost_exchange.hpp` |
| `drivers/partitioning/spmv_{rcm,fennel,block}_nccl_{acc,cusparse}.cu` | 6 driver mains (NCCL transport only) |

---

## Build

Built automatically by the root `bash build.sh` (no separate build step).
All 6 binaries land in the project root's `bin/` alongside every other driver.

---

## Drivers at a glance

| Binary | Partitioning | Kernel |
|---|---|---|
| `spmv_rcm_nccl_cusparse` | RCM | cuSPARSE |
| `spmv_rcm_nccl_acc` | RCM | ACC |
| `spmv_fennel_nccl_cusparse` | Fennel/LDG | cuSPARSE |
| `spmv_fennel_nccl_acc` | Fennel/LDG | ACC |
| `spmv_block_nccl_cusparse` | Block | cuSPARSE |
| `spmv_block_nccl_acc` | Block | ACC |

Same usage and `RESULT` line format as every other driver (see the project
root `README.md`): `mpirun -n <P> ./bin/<driver> <matrix.mtx>`, with
`comm=rcm_nccl` / `fennel_nccl` / `block_nccl` identifying the strategy.

---

## Running

All commands below run from the project root (`sbatch`'s working directory
is the directory it was invoked from).

### Small matrices (`matrices/`, P=1..4, one job per matrix)

```bash
sbatch bonus_partitioning_strategies/sbatch/rcm_small_matrices.sh matrices/<name>.mtx [REPS]
sbatch bonus_partitioning_strategies/sbatch/rcm_cusparse_small_matrices.sh matrices/<name>.mtx [REPS]
sbatch bonus_partitioning_strategies/sbatch/fennel_small_matrices.sh matrices/<name>.mtx [REPS]
sbatch bonus_partitioning_strategies/sbatch/fennel_cusparse_small_matrices.sh matrices/<name>.mtx [REPS]
sbatch bonus_partitioning_strategies/sbatch/block_small_matrices.sh matrices/<name>.mtx [REPS]
sbatch bonus_partitioning_strategies/sbatch/block_cusparse_small_matrices.sh matrices/<name>.mtx [REPS]

# submit one of the above for all 10 small matrices at once
bash bonus_partitioning_strategies/sbatch/submit_small_matrices.sh <script> [REPS]
```

### Big matrices (`big_matrices/`, one matrix per job, sweeps P=1..4 internally, `REPS` defaults to 1)

```bash
sbatch bonus_partitioning_strategies/sbatch/large_matrices_night.sh big_matrices/<name>.mtx [REPS] [DRIVER]
# DRIVER defaults to spmv_rcm_nccl_cusparse; valid: spmv_{rcm,fennel,block}_nccl_{acc,cusparse}

# submit all 10 big matrices for one driver at once
bash bonus_partitioning_strategies/sbatch/submit_large_matrices_night.sh [REPS] [DRIVER]
```

### Weak scaling (all 6 drivers, one job per P)

Needs `matrices_synthetic/weak_P<P>.mtx` — generate first with
`python scripts/gen_weak_matrix.py --all` from the project root:

```bash
sbatch bonus_partitioning_strategies/sbatch/weak_scaling_partitioned.sh <P> [REPS] [MATRIX]
bash bonus_partitioning_strategies/sbatch/submit_weak_scaling_partitioned.sh [REPS] [weak|weak_sparse]
```

`weak_p4_interleaved_check.sh` is a diagnostic, not part of the normal
sweep: cyclic vs. RCM/Fennel/Block interleaved in the same job at P=4, used
to tell a real partitioning effect apart from run-to-run NCCL timing noise.

---

## Analysis

Every script below is run from the project root and resolves its own paths
relative to its own file location, so it works regardless of the current
shell's working directory:

| Script | Reads | Writes |
|---|---|---|
| `analyze_rcm.py` | `outputs/rcm/rcm-*.out` | `outputs/csv/rcm_vs_cyclic_summary.csv` |
| `analyze_fennel.py` | `outputs/fennel/fennel-*.out` | `outputs/csv/fennel_vs_cyclic_summary.csv` |
| `analyze_block.py` | `outputs/block/block-*.out` | `outputs/csv/block_vs_cyclic_summary.csv` |
| `analyze_kernel_isolation.py` | all 3 prototypes, both kernels | `outputs/csv/kernel_isolation_summary.csv` |
| `analyze_weak_partitioned.py` | `outputs/weak_scaling/weak_partitioned-*.out` | `outputs/csv/weak_partitioned_summary.csv` |

Each `*_vs_cyclic` script cross-references the root `outputs/csv/strong_scaling_summary.csv`
for the baseline cyclic driver's numbers at the same matrix/P, so a direct,
one-variable-changed speedup can be reported. `analyze_weak_partitioned.py`
takes an optional `--cyclic-summary` flag to compare against the low-degree
synthetic family instead of the default saturated one (see the script's own
docstring).

```bash
python bonus_partitioning_strategies/analyze_rcm.py
python bonus_partitioning_strategies/analyze_fennel.py
python bonus_partitioning_strategies/analyze_block.py
python bonus_partitioning_strategies/analyze_kernel_isolation.py
python bonus_partitioning_strategies/analyze_weak_partitioned.py
```

---

## Output layout

```
outputs/rcm/, outputs/fennel/, outputs/block/     # small-matrix .out/.err logs
outputs/large_matrices/                           # big-matrix .out/.err logs
outputs/weak_scaling/                             # weak-scaling .out/.err logs
outputs/csv/                                      # *_summary.csv tables (above)
```
