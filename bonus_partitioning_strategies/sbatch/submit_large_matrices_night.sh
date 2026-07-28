#!/bin/bash
# Submit large_matrices_night.sh for every big matrix, one driver.
# Usage: bash bonus_partitioning_strategies/sbatch/submit_large_matrices_night.sh [REPS] [DRIVER]
set -euo pipefail

REPS="${1:-1}"
DRIVER="${2:-spmv_rcm_nccl_cusparse}"

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
    echo "Submitting ${m} (REPS=${REPS}, DRIVER=${DRIVER}) ..."
    sbatch bonus_partitioning_strategies/sbatch/large_matrices_night.sh "${m}" "${REPS}" "${DRIVER}"
done
