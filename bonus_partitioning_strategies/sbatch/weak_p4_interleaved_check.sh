#!/bin/bash
# Same-job A/B check: cyclic ghost_nccl vs RCM/Fennel/Block, interleaved, P=4.
# Usage: sbatch bonus_partitioning_strategies/sbatch/weak_p4_interleaved_check.sh [REPS] [MATRIX]
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_weak_p4_ab
#SBATCH --output=bonus_partitioning_strategies/outputs/weak_scaling/weak_p4_interleaved-%j.out
#SBATCH --error=bonus_partitioning_strategies/outputs/weak_scaling/weak_p4_interleaved-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

REPS="${1:-5}"
MATRIX="${2:-matrices_synthetic/weak_P4.mtx}"

if [[ ! -f "${MATRIX}" ]]; then
    echo "ERROR: ${MATRIX} not found." >&2
    exit 1
fi

mkdir -p bonus_partitioning_strategies/outputs/weak_scaling

DRIVERS=(spmv_ghost_nccl_acc spmv_ghost_nccl_cusparse \
         spmv_rcm_nccl_acc spmv_rcm_nccl_cusparse \
         spmv_fennel_nccl_acc spmv_fennel_nccl_cusparse \
         spmv_block_nccl_acc spmv_block_nccl_cusparse)

echo "=== weak-scaling P=4 same-job A/B check (cyclic vs RCM/Fennel/Block) ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "Reps:     ${REPS}"
echo "Drivers:  ${DRIVERS[*]}"
echo ""

for rep in $(seq 1 "${REPS}"); do
    for driver in "${DRIVERS[@]}"; do
        if [[ ! -x "./bin/${driver}" ]]; then
            echo "--- rep=${rep}/${REPS} driver=${driver}: SKIPPED (binary not built) ---"
            continue
        fi
        echo "--- rep=${rep}/${REPS} driver=${driver} matrix=${MATRIX} ---"
        mpirun -n 4 --mca mpi_common_cuda_register_memory 0 "./bin/${driver}" "${MATRIX}"
        echo ""
    done
done

echo "=== Job done ==="
