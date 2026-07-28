#!/bin/bash
# Usage: bash distributed_matrix_read/sbatch/submit_big_matrices_mtx_read.sh [REPS]
#   REPS defaults to 1
set -euo pipefail

REPS="${1:-1}"

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
    sbatch distributed_matrix_read/sbatch/mtx_read_benchmark_large.sh "${m}" "${REPS}"
done
