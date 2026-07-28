#!/bin/bash
# RCM + cuSPARSE, one matrix, P=1..4.
# Usage: sbatch bonus_partitioning_strategies/sbatch/rcm_cusparse_small_matrices.sh <path-to-mtx> [REPS]
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_rcm_cus
#SBATCH --output=bonus_partitioning_strategies/outputs/rcm/rcm_cusparse-%j.out
#SBATCH --error=bonus_partitioning_strategies/outputs/rcm/rcm_cusparse-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch bonus_partitioning_strategies/sbatch/rcm_cusparse_small_matrices.sh <path-to-mtx> [REPS]}"
REPS="${2:-3}"

exec bash "bonus_partitioning_strategies/sbatch/_run_small_matrix.sh" rcm cusparse "${MATRIX}" "${REPS}"
