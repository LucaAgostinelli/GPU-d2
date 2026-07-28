#!/bin/bash
# Submit one job per (matrix, P) or per P, depending on the target script.
# Usage: bash sbatch/submit_all_matrices.sh [sbatch_script] [REPS]
#   sbatch_script defaults to sbatch/strong_scaling.sh, REPS defaults to 3
set -euo pipefail

SCRIPT="${1:-sbatch/strong_scaling.sh}"
REPS="${2:-3}"

if [[ ! -f "${SCRIPT}" ]]; then
    echo "ERROR: ${SCRIPT} not found in $(pwd)." >&2
    exit 1
fi

mkdir -p outputs/baseline outputs/strong_scaling outputs/weak_scaling

job_ids=()

case "$(basename "${SCRIPT}")" in
    weak_scaling.sh)
        echo "Submitting ${SCRIPT} for P=1..4 (REPS=${REPS}) ..."
        echo ""
        for P in 1 2 3 4; do
            echo -n "  P=${P} ... "
            out="$(sbatch "${SCRIPT}" "${P}" "${REPS}")"
            echo "${out}"
            job_ids+=("$(echo "${out}" | awk '{print $NF}')")
        done
        ;;
    cpu_baseline.sh)
        echo "Submitting ${SCRIPT} for every matrices/*.mtx (REPS=${REPS}) ..."
        echo ""
        for MATRIX in matrices/*.mtx; do
            echo -n "  ${MATRIX} ... "
            out="$(sbatch "${SCRIPT}" "${MATRIX}" "${REPS}")"
            echo "${out}"
            job_ids+=("$(echo "${out}" | awk '{print $NF}')")
        done
        ;;
    *)
        echo "Submitting ${SCRIPT} for every matrices/*.mtx x P=1..4 (REPS=${REPS}) ..."
        echo ""
        for MATRIX in matrices/*.mtx; do
            for P in 1 2 3 4; do
                echo -n "  ${MATRIX} P=${P} ... "
                out="$(sbatch "${SCRIPT}" "${MATRIX}" "${P}" "${REPS}")"
                echo "${out}"
                job_ids+=("$(echo "${out}" | awk '{print $NF}')")
            done
        done
        ;;
esac

echo ""
echo "Submitted ${#job_ids[@]} jobs: ${job_ids[*]}"
echo "Check status with:  squeue -u \$USER"
