#!/bin/bash
# =============================================================================
# Runs bench_mtx_read for one matrix
# at one P, comparing two ways to get from a .mtx file on disk to a
# distributed LocalCSR shard on every rank:
#   - "serial":  rank 0 reads the whole file (read_mtx) and Scatterv's
#     pre-built CSR shards.
#   - "chunked": every rank independently opens the file, parses the header
#     itself, and reads its own line-aligned byte range with a plain
#     ifstream::read then redistributes via MPI_Alltoallv, same as "serial"'s underlying rule.
#
# Usage:
#   sbatch distributed_matrix_read/sbatch/mtx_read_benchmark.sh <path-to-mtx> <P> [REPS]
#   (P must be 1, 2, 3, or 4; REPS defaults to 3)
# =============================================================================
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_mtx_read
#SBATCH --output=distributed_matrix_read/outputs/mtx_read-%j.out
#SBATCH --error=distributed_matrix_read/outputs/mtx_read-%j.err

set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

MATRIX="${1:?Usage: sbatch distributed_matrix_read/sbatch/mtx_read_benchmark.sh <path-to-mtx> <P> [REPS]}"
P="${2:?Usage: sbatch distributed_matrix_read/sbatch/mtx_read_benchmark.sh <path-to-mtx> <P> [REPS]}"
REPS="${3:-3}"

case "${P}" in
    1|2|3|4) ;;
    *)
        echo "ERROR: P must be 1, 2, 3, or 4 (got ${P})." >&2
        exit 1
        ;;
esac

if [[ ! -x "./bin/bench_mtx_read" ]]; then
    echo "ERROR: ./bin/bench_mtx_read not found (build.sh hasn't been rerun since it was added?)." >&2
    exit 1
fi

mkdir -p distributed_matrix_read/outputs

echo "=== distributed matrix read benchmark ==="
echo "Node:    $(hostname)"
echo "Date:    $(date)"
echo "Matrix:  ${MATRIX}"
echo "P:       ${P}"
echo "Reps:    ${REPS}"
echo ""

time mpirun -n "${P}" "./bin/bench_mtx_read" "${MATRIX}" "${REPS}"

echo ""
echo "=== Job done ==="
