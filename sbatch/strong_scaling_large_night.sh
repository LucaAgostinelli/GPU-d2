#!/bin/bash
# One matrix, sweeps P=1..4, ONE driver, meant to run unattended overnight (edu-medium).
# Usage: sbatch sbatch/strong_scaling_large_night.sh <path-to-mtx> [REPS] [DRIVER]
#   REPS defaults to 1; DRIVER defaults to spmv_ghost_nccl_acc; spmv_2d_* skips P=3
# After: python scripts/analyze_strong_scaling.py outputs/large_matrices/strong_large_night-*.out
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_strong_large_night
#SBATCH --output=outputs/large_matrices/strong_large_night-%j.out
#SBATCH --error=outputs/large_matrices/strong_large_night-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch sbatch/strong_scaling_large_night.sh <path-to-mtx> [REPS] [DRIVER]}"
REPS="${2:-1}"
DRIVER="${3:-spmv_ghost_nccl_acc}"

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

if [[ ! -x "./bin/${DRIVER}" ]]; then
    echo "ERROR: ./bin/${DRIVER} not found (not built, or NCCL wasn't found at build time)." >&2
    exit 1
fi

mkdir -p outputs/large_matrices

echo "=== strong scaling (large matrix, full P sweep, single driver) ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "Reps:     ${REPS}"
echo "Driver:   ${DRIVER}"
echo ""

for P in 1 2 3 4; do
    if [[ "${P}" == "3" && "${DRIVER}" == spmv_2d* ]]; then
        echo "--- P=3 driver=${DRIVER}: SKIPPED (no non-degenerate process grid at P=3) ---"
        continue
    fi
    for rep in $(seq 1 "${REPS}"); do
        echo "--- P=${P} rep=${rep}/${REPS} driver=${DRIVER} ---"
        time mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 "./bin/${DRIVER}" "${MATRIX}"
        echo ""
    done
done

echo "=== Job done ==="
