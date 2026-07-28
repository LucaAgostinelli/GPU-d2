#!/bin/bash
# A/B check: does --mca mpi_common_cuda_register_memory 0 suppress the cuMemHostRegister/smcuda fallback?
# Runs the same driver+matrix+P twice (default vs. flag set) so the two .out/.err blocks are diffable.
# Usage: sbatch sbatch/mca_register_memory_check.sh [driver] [path-to-mtx] [P] [REPS]
#   Defaults: driver=spmv_ghost_cusparse matrix=matrices/cit-Patents.mtx P=4 REPS=8
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_mca_check
#SBATCH --output=outputs/mca_check/mca_check-%j.out
#SBATCH --error=outputs/mca_check/mca_check-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

DRIVER="${1:-spmv_ghost_cusparse}"
MATRIX="${2:-matrices/cit-Patents.mtx}"
P="${3:-4}"
REPS="${4:-8}"

mkdir -p outputs/mca_check

echo "=== MCA register-memory A/B check ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Driver:   ${DRIVER}"
echo "Matrix:   ${MATRIX}"
echo "P:        ${P}"
echo "Reps:     ${REPS}"
echo ""

if [[ ! -x "./bin/${DRIVER}" ]]; then
    echo "ERROR: ./bin/${DRIVER} not built." >&2
    exit 1
fi

echo "=== WITHOUT --mca mpi_common_cuda_register_memory 0 (default) ===" >&2
echo "=== WITHOUT --mca mpi_common_cuda_register_memory 0 (default) ==="
for rep in $(seq 1 "${REPS}"); do
    echo "--- default rep=${rep}/${REPS} ---"
    mpirun -n "${P}" "./bin/${DRIVER}" "${MATRIX}"
    echo ""
done

echo "=== WITH --mca mpi_common_cuda_register_memory 0 ==="
for rep in $(seq 1 "${REPS}"); do
    echo "--- register_memory=0 rep=${rep}/${REPS} ---"
    mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 \
        "./bin/${DRIVER}" "${MATRIX}"
    echo ""
done

echo "=== Job done ==="
echo ""
echo "Compare stderr fallback counts:"
echo "  grep -c cuMemHostRegister outputs/mca_check/mca_check-${SLURM_JOB_ID}.err || true"
echo "Compare timings (RESULT lines) between the two blocks above in:"
echo "  outputs/mca_check/mca_check-${SLURM_JOB_ID}.out"
