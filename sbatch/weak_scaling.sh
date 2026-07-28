#!/bin/bash
# Fixed P, its own synthetic matrix (matrices_synthetic/weak_P<P>.mtx), all 9 GPU/MPI drivers.
# Usage: sbatch sbatch/weak_scaling.sh <P> [REPS] [MATRIX]
#   P must be 1-4; REPS defaults to 3
# After: python scripts/analyze_weak_scaling.py outputs/weak_scaling/weak-*.out
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_weak
#SBATCH --output=outputs/weak_scaling/weak-%j.out
#SBATCH --error=outputs/weak_scaling/weak-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

P="${1:?Usage: sbatch sbatch/weak_scaling.sh <P> [REPS] [MATRIX]}"
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

mkdir -p outputs/weak_scaling

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

echo "=== weak scaling ==="
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
