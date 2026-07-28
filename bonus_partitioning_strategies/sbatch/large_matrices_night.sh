#!/bin/bash
# One big matrix, sweeps P=1..4, one prototype driver, unattended overnight run.
# Usage: sbatch bonus_partitioning_strategies/sbatch/large_matrices_night.sh <path-to-mtx> [REPS] [DRIVER]
#   REPS defaults to 1; DRIVER defaults to spmv_rcm_nccl_cusparse
#   Valid DRIVER: spmv_{rcm,fennel,block}_nccl_{acc,cusparse}
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_ip_large_night
#SBATCH --output=bonus_partitioning_strategies/outputs/large_matrices/large_night-%j.out
#SBATCH --error=bonus_partitioning_strategies/outputs/large_matrices/large_night-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch bonus_partitioning_strategies/sbatch/large_matrices_night.sh <path-to-mtx> [REPS] [DRIVER]}"
REPS="${2:-1}"
DRIVER="${3:-spmv_rcm_nccl_cusparse}"

case "${DRIVER}" in
    spmv_rcm_nccl_acc|spmv_rcm_nccl_cusparse| \
    spmv_fennel_nccl_acc|spmv_fennel_nccl_cusparse| \
    spmv_block_nccl_acc|spmv_block_nccl_cusparse)
        ;;
    *)
        echo "ERROR: unknown DRIVER '${DRIVER}'." >&2
        exit 1
        ;;
esac

if [[ ! -x "./bin/${DRIVER}" ]]; then
    echo "ERROR: ./bin/${DRIVER} not found (not built, or NCCL wasn't found at build time?)." >&2
    exit 1
fi

mkdir -p bonus_partitioning_strategies/outputs/large_matrices

echo "=== bonus partitioning strategies: large-matrix check (full P sweep, single driver) ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "Reps:     ${REPS}"
echo "Driver:   ${DRIVER}"
echo ""

for P in 1 2 3 4; do
    for rep in $(seq 1 "${REPS}"); do
        echo "--- P=${P} rep=${rep}/${REPS} driver=${DRIVER} ---"
        time mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 "./bin/${DRIVER}" "${MATRIX}"
        echo ""
    done
done

echo "=== Job done ==="
