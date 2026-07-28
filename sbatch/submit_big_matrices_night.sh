#!/bin/bash
# Submit one strong_scaling_large_night.sh job per big matrix (independent jobs, not one loop).
# Usage: bash sbatch/submit_big_matrices_night.sh [REPS] [DRIVER]
#   REPS defaults to 1, DRIVER defaults to spmv_ghost_nccl_acc
set -euo pipefail

REPS="${1:-1}"
DRIVER="${2:-spmv_ghost_nccl_acc}"

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
    sbatch sbatch/strong_scaling_large_night.sh "${m}" "${REPS}" "${DRIVER}"
done
