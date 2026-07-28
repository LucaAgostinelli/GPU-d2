#!/bin/bash
# Usage:
#   sbatch distributed_matrix_read/sbatch/mtx_read_benchmark_large.sh <path-to-mtx> [REPS]
# =========================================================================================
#SBATCH --partition=edu-medium
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_mtx_read_large
#SBATCH --output=distributed_matrix_read/outputs/mtx_read-%j.out
#SBATCH --error=distributed_matrix_read/outputs/mtx_read-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch distributed_matrix_read/sbatch/mtx_read_benchmark_large.sh <path-to-mtx> [REPS]}"
REPS="${2:-1}"

if [[ ! -x "./bin/bench_mtx_read" ]]; then
    echo "ERROR: ./bin/bench_mtx_read not found (build.sh hasn't been rerun since it was added?)." >&2
    exit 1
fi

mkdir -p distributed_matrix_read/outputs

echo "=== distributed matrix read benchmark (large matrix, full P sweep) ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Matrix:  ${MATRIX}"
echo "Reps:    ${REPS}"
echo ""

for P in 1 2 3 4; do
    echo "--- P=${P} ---"
    time mpirun -n "${P}" "./bin/bench_mtx_read" "${MATRIX}" "${REPS}"
    echo ""
done

echo "=== Job done ==="
