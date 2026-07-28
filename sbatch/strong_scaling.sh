#!/bin/bash
# Fixed matrix, fixed P, all 9 GPU/MPI drivers (2D drivers skipped at P=3 -- no valid process grid).
# Usage: sbatch sbatch/strong_scaling.sh <path-to-mtx> <P> [REPS]
#   P must be 1-4; REPS defaults to 3
# After: python scripts/analyze_strong_scaling.py outputs/strong_scaling/strong-*.out
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_strong
#SBATCH --output=outputs/strong_scaling/strong-%j.out
#SBATCH --error=outputs/strong_scaling/strong-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch sbatch/strong_scaling.sh <path-to-mtx> <P> [REPS]}"
P="${2:?Usage: sbatch sbatch/strong_scaling.sh <path-to-mtx> <P> [REPS]}"
REPS="${3:-3}"

case "${P}" in
    1|2|3|4) ;;
    *)
        echo "ERROR: P must be 1, 2, 3, or 4 (got ${P})." >&2
        exit 1
        ;;
esac

mkdir -p outputs/strong_scaling

DRIVERS_ALL=(spmv_bcast spmv_ghost_cusparse spmv_ghost_acc \
             spmv_ghost_nccl_cusparse spmv_ghost_nccl_acc \
             spmv_2d_cusparse spmv_2d_acc \
             spmv_2d_nccl_cusparse spmv_2d_nccl_acc)
DRIVERS_NO_2D=(spmv_bcast spmv_ghost_cusparse spmv_ghost_acc \
               spmv_ghost_nccl_cusparse spmv_ghost_nccl_acc)

if [[ "${P}" == "3" ]]; then
    DRIVERS=("${DRIVERS_NO_2D[@]}")
else
    DRIVERS=("${DRIVERS_ALL[@]}")
fi

echo "=== strong scaling ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "P:        ${P}"
echo "Reps:     ${REPS}"
echo ""

for rep in $(seq 1 "${REPS}"); do
    for driver in "${DRIVERS[@]}"; do
        if [[ ! -x "./bin/${driver}" ]]; then
            echo "--- P=${P} rep=${rep}/${REPS} driver=${driver}: SKIPPED (binary not built) ---"
            continue
        fi
        echo "--- P=${P} rep=${rep}/${REPS} driver=${driver} ---"
        mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 "./bin/${driver}" "${MATRIX}"
        echo ""
    done
done

echo "=== Job done ==="
