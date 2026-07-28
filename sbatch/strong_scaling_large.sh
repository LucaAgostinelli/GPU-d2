#!/bin/bash
# Fixed matrix, fixed P, ONE driver only -- companion to strong_scaling.sh for matrices too large for a full 9-driver sweep.
# Usage: sbatch sbatch/strong_scaling_large.sh <path-to-mtx> <P> [REPS] [DRIVER]
#   P must be 1-4; REPS defaults to 1; DRIVER defaults to spmv_ghost_nccl_acc
# After: python scripts/analyze_strong_scaling.py outputs/large_matrices/strong_large-*.out
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_strong_large
#SBATCH --output=outputs/large_matrices/strong_large-%j.out
#SBATCH --error=outputs/large_matrices/strong_large-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch sbatch/strong_scaling_large.sh <path-to-mtx> <P> [REPS] [DRIVER]}"
P="${2:?Usage: sbatch sbatch/strong_scaling_large.sh <path-to-mtx> <P> [REPS] [DRIVER]}"
REPS="${3:-1}"
DRIVER="${4:-spmv_ghost_nccl_acc}"

case "${P}" in
    1|2|3|4) ;;
    *)
        echo "ERROR: P must be 1, 2, 3, or 4 (got ${P})." >&2
        exit 1
        ;;
esac

case "${DRIVER}" in
    spmv_bcast|spmv_bcast_acc|spmv_ghost_cusparse|spmv_ghost_acc| \
    spmv_ghost_nccl_cusparse|spmv_ghost_nccl_acc| \
    spmv_2d_cusparse|spmv_2d_acc| \
    spmv_2d_nccl_cusparse|spmv_2d_nccl_acc)
        ;;
    *)
        echo "ERROR: unknown DRIVER '${DRIVER}'." >&2
        exit 1
        ;;
esac

if [[ "${P}" == "3" && "${DRIVER}" == spmv_2d* ]]; then
    echo "ERROR: spmv_2d_* drivers have no non-degenerate process grid at P=3." >&2
    exit 1
fi

if [[ ! -x "./bin/${DRIVER}" ]]; then
    echo "ERROR: ./bin/${DRIVER} not found (not built, or NCCL wasn't found at build time)." >&2
    exit 1
fi

mkdir -p outputs/large_matrices

echo "=== strong scaling (large matrix, single driver) ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "P:        ${P}"
echo "Reps:     ${REPS}"
echo "Driver:   ${DRIVER}"
echo ""

for rep in $(seq 1 "${REPS}"); do
    echo "--- P=${P} rep=${rep}/${REPS} driver=${DRIVER} ---"
    time mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 "./bin/${DRIVER}" "${MATRIX}"
    echo ""
done

echo "=== Job done ==="
