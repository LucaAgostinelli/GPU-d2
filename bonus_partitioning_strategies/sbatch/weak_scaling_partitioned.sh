#!/bin/bash
# Weak scaling for RCM/Fennel/Block, both kernels (6 drivers), one job per P.
# Usage: sbatch bonus_partitioning_strategies/sbatch/weak_scaling_partitioned.sh <P> [REPS] [MATRIX]
#   P must be 1-4; MATRIX defaults to matrices_synthetic/weak_P<P>.mtx
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_weak_part
#SBATCH --output=bonus_partitioning_strategies/outputs/weak_scaling/weak_partitioned-%j.out
#SBATCH --error=bonus_partitioning_strategies/outputs/weak_scaling/weak_partitioned-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

P="${1:?Usage: sbatch bonus_partitioning_strategies/sbatch/weak_scaling_partitioned.sh <P> [REPS] [MATRIX]}"
REPS="${2:-3}"
MATRIX="${3:-matrices_synthetic/weak_P${P}.mtx}"

case "${P}" in
    1|2|3|4) ;;
    *)
        echo "ERROR: P must be 1, 2, 3, or 4 (got ${P})." >&2
        exit 1
        ;;
esac

if [[ ! -f "${MATRIX}" ]]; then
    echo "ERROR: ${MATRIX} not found." >&2
    echo "Generate locally with 'python scripts/gen_weak_matrix.py --all' (or" >&2
    echo "gen_weak_matrix_sparse.py --all for the low-degree variant) and scp it over." >&2
    exit 1
fi

mkdir -p bonus_partitioning_strategies/outputs/weak_scaling

DRIVERS=(spmv_rcm_nccl_acc spmv_rcm_nccl_cusparse \
         spmv_fennel_nccl_acc spmv_fennel_nccl_cusparse \
         spmv_block_nccl_acc spmv_block_nccl_cusparse)

echo "=== bonus partitioning strategies: weak scaling ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "P:        ${P}"
echo "Matrix:   ${MATRIX}"
echo "Reps:     ${REPS}"
echo ""

for rep in $(seq 1 "${REPS}"); do
    for driver in "${DRIVERS[@]}"; do
        if [[ ! -x "./bin/${driver}" ]]; then
            echo "--- P=${P} rep=${rep}/${REPS} driver=${driver}: SKIPPED (binary not built) ---"
            continue
        fi
        echo "--- P=${P} rep=${rep}/${REPS} driver=${driver} matrix=${MATRIX} ---"
        mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 "./bin/${driver}" "${MATRIX}"
        echo ""
    done
done

echo "=== Job done ==="
