#!/bin/bash
# Shared implementation for the 6 small-matrix entry points (not sbatch'd directly).
# Usage: _run_small_matrix.sh <prototype:rcm|fennel|block> <kernel:acc|cusparse> <path-to-mtx> [REPS]
set -euo pipefail

PROTOTYPE="${1:?prototype required: rcm, fennel, or block}"
KERNEL="${2:?kernel required: acc or cusparse}"
MATRIX="${3:?path-to-mtx required}"
REPS="${4:-3}"

case "${PROTOTYPE}" in
    rcm|fennel|block) ;;
    *)
        echo "ERROR: unknown prototype '${PROTOTYPE}' (expected rcm, fennel, or block)." >&2
        exit 1
        ;;
esac

case "${KERNEL}" in
    acc|cusparse) ;;
    *)
        echo "ERROR: unknown kernel '${KERNEL}' (expected acc or cusparse)." >&2
        exit 1
        ;;
esac

DRIVER="spmv_${PROTOTYPE}_nccl_${KERNEL}"

if [[ ! -x "./bin/${DRIVER}" ]]; then
    echo "ERROR: ./bin/${DRIVER} not built (NCCL not found at build time?)." >&2
    exit 1
fi

mkdir -p "bonus_partitioning_strategies/outputs/${PROTOTYPE}"

echo "=== bonus partitioning strategies: ${PROTOTYPE} (${KERNEL}) small-matrix check ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Matrix:   ${MATRIX}"
echo "Reps:     ${REPS}"
echo "Driver:   ${DRIVER}"
echo ""

for rep in $(seq 1 "${REPS}"); do
    for P in 1 2 3 4; do
        echo "--- P=${P} rep=${rep}/${REPS} driver=${DRIVER} ---"
        mpirun -n "${P}" --mca mpi_common_cuda_register_memory 0 \
            "./bin/${DRIVER}" "${MATRIX}"
        echo ""
    done
done

echo "=== Job done ==="
