#!/bin/bash
# =============================================================================
# Usage:
#   bash build.sh          # standard build
#   bash build.sh clean    # wipe build/ and bin/, then rebuild
# =============================================================================
set -euo pipefail

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2
module load CMake/3.26.3-GCCcore-12.3.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "${1:-}" == "clean" ]]; then
    echo "==> Cleaning build/ and bin/ ..."
    rm -rf build bin
fi

mkdir -p build bin
cd build

echo "==> Running cmake ..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=80

echo "==> Building ($(nproc) cores) ..."
make -j"$(nproc)" VERBOSE=0

echo ""
echo "=== Build complete -- binaries in ${SCRIPT_DIR}/bin/ ==="
