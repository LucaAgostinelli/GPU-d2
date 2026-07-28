#!/bin/bash
# Submit any small-matrix entry-point script for all 10 small matrices.
# Usage: bash bonus_partitioning_strategies/sbatch/submit_small_matrices.sh <target_script> [REPS]
set -euo pipefail

SCRIPT="${1:?Usage: bash bonus_partitioning_strategies/sbatch/submit_small_matrices.sh <target_script> [REPS]}"
REPS="${2:-3}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: ${SCRIPT} not found in $(pwd)." >&2
    exit 1
fi

mkdir -p bonus_partitioning_strategies/outputs

job_ids=()

echo "Submitting ${SCRIPT} for every matrices/*.mtx (REPS=${REPS}) ..."
echo ""
for MATRIX in matrices/*.mtx; do
    echo -n "  ${MATRIX} ... "
    out="$(sbatch "${SCRIPT}" "${MATRIX}" "${REPS}")"
    echo "${out}"
    job_ids+=("$(echo "${out}" | awk '{print $NF}')")
done

echo ""
echo "Submitted ${#job_ids[@]} jobs: ${job_ids[*]}"
echo "Check status with:  squeue -u \$USER"
