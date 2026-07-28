#!/bin/bash
# One cpu_baseline.sh job per big matrix, overriding its partition/time
# (big matrices' serial read dominates wall time).
# Usage: bash sbatch/submit_big_matrices_baseline.sh [REPS]  (REPS defaults to 10)
set -euo pipefail

REPS="${1:-10}"

MATRICES=(
    big_matrices/arabic-2005.mtx
    big_matrices/com-Orkut.mtx
    big_matrices/europe_osm.mtx
    big_matrices/kmer_V1r.mtx
    big_matrices/mawi_201512020330.mtx
    big_matrices/mycielskian19.mtx
    big_matrices/nlpkkt240.mtx
    big_matrices/Queen_4147.mtx
    big_matrices/uk-2005.mtx
    big_matrices/webbase-2001.mtx
)

for m in "${MATRICES[@]}"; do
    if [[ ! -f "${m}" ]]; then
        echo "WARNING: ${m} not found, skipping." >&2
        continue
    fi
    echo "Submitting ${m} (REPS=${REPS}) ..."
    sbatch --partition=edu-medium --time=02:00:00 sbatch/cpu_baseline.sh "${m}" "${REPS}"
done
