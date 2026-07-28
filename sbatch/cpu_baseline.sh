#!/bin/bash
# Serial single-core CPU reference, no GPU/MPI.
# Usage: sbatch sbatch/cpu_baseline.sh <path-to-mtx> [reps]
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_baseline
#SBATCH --output=outputs/baseline/baseline-%j.out
#SBATCH --error=outputs/baseline/baseline-%j.err

set -euo pipefail

MATRIX="${1:?Usage: sbatch sbatch/cpu_baseline.sh <path-to-mtx> [reps]}"
REPS="${2:-10}"

mkdir -p outputs/baseline

echo "=== CPU baseline ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "Reps:     ${REPS}"
echo ""

time ./bin/spmv_cpu_baseline "${MATRIX}" "${REPS}"

echo ""
echo "=== Job done ==="
