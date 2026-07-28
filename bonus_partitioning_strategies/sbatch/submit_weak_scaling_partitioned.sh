#!/bin/bash
# Submit weak_scaling_partitioned.sh for P=1..4.
# Usage: bash bonus_partitioning_strategies/sbatch/submit_weak_scaling_partitioned.sh [REPS] [FAMILY]
#   FAMILY: weak (default) or weak_sparse
set -euo pipefail

REPS="${1:-3}"
FAMILY="${2:-weak}"

mkdir -p bonus_partitioning_strategies/outputs/weak_scaling

job_ids=()

echo "Submitting weak_scaling_partitioned.sh for P=1,2,3,4 (REPS=${REPS}, family=${FAMILY}) ..."
echo ""
for P in 1 2 3 4; do
    MATRIX="matrices_synthetic/${FAMILY}_P${P}.mtx"
    if [[ "${FAMILY}" == "weak" ]]; then
        MATRIX="matrices_synthetic/weak_P${P}.mtx"
    fi
    echo -n "  P=${P} (${MATRIX}) ... "
    out="$(sbatch bonus_partitioning_strategies/sbatch/weak_scaling_partitioned.sh "${P}" "${REPS}" "${MATRIX}")"
    echo "${out}"
    job_ids+=("$(echo "${out}" | awk '{print $NF}')")
done

echo ""
echo "Submitted ${#job_ids[@]} jobs: ${job_ids[*]}"
echo "Check status with:  squeue -u \$USER"
echo "After they complete:  python bonus_partitioning_strategies/analyze_weak_partitioned.py"
