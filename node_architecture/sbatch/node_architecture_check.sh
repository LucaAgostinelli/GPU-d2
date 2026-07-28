#!/bin/bash
# =============================================================================
# diagnostic: full hardware/software profile of the compute node this project runs on (edu01)
#
# This script collects everything relevant in one place, in one run:
#   - CPU: model, socket/core/thread count, NUMA layout, cache sizes
#   - Memory: total RAM, per-NUMA-node memory (bounds "largest manageable matrix size")
#   - GPU: model, per-GPU memory, compute capability, SM count, memory bus
#     width/theoretical bandwidth (roofline context for GFLOPS numbers),
#     PCIe generation/width both *capable* and *currently negotiated*
#   - GPU interconnect topology + NUMA/CPU affinity and PCIe ACS status
#   - Network fabric: confirms this is a single-node, 4-GPU job (no
#     multi-node interconnect/IB fabric is actually exercised by anything in this project)
#   - Software stack versions actually in use: CUDA/driver, Open MPI, NCCL,
#     GCC, CMake
#   - SLURM's own view of the node (partition membership, allocatable
#     resources, any node-level features/gres config)
#   - Filesystem: available space where matrices/outputs live
#
# Usage:
#   sbatch node_architecture/sbatch/node_architecture_check.sh
# =============================================================================
#SBATCH --partition=edu-short
#SBATCH --account=gpu.computing26
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --nodelist=edu01

#SBATCH --job-name=d2_node_arch
#SBATCH --output=node_architecture/outputs/node_arch-%j.out
#SBATCH --error=node_architecture/outputs/node_arch-%j.err

set -uo pipefail   # not -e: many diagnostic commands are allowed to fail/be absent

module load CUDA/12.3.2
module load OpenMpi/4.1.5-CUDA-12.3.2

mkdir -p node_architecture/outputs

section() {
    echo ""
    echo "=============================================================================="
    echo "== $1"
    echo "=============================================================================="
}

run() {
    # run <description> <command...> -- prints the command, runs it, tolerates failure
    local desc="$1"; shift
    echo "--- $desc ---"
    if ! "$@" 2>&1; then
        echo "(command failed or not available: $*)"
    fi
    echo ""
}

echo "=== node architecture profile ==="
echo "Node:     $(hostname)"
echo "Date:     $(date)"
echo "Job ID:   ${SLURM_JOB_ID:-<none>}"

# -----------------------------------------------------------------------------
section "OS / KERNEL"
# -----------------------------------------------------------------------------
run "uname -a"            uname -a
run "/etc/os-release"     cat /etc/os-release
run "uptime"               uptime

# -----------------------------------------------------------------------------
section "CPU"
# -----------------------------------------------------------------------------
run "lscpu (sockets, cores/socket, threads/core, NUMA nodes, cache sizes, flags)" lscpu
run "nproc --all"          nproc --all

# -----------------------------------------------------------------------------
section "NUMA topology and per-node memory"
# -----------------------------------------------------------------------------
run "numactl --hardware"   numactl --hardware
run "lscpu NUMA lines only" bash -c "lscpu | grep -i numa"

# -----------------------------------------------------------------------------
section "MEMORY"
# -----------------------------------------------------------------------------
run "free -h"               free -h
run "/proc/meminfo (MemTotal/HugePages)" bash -c "grep -E 'MemTotal|HugePages' /proc/meminfo"

# -----------------------------------------------------------------------------
section "GPU (nvidia-smi)"
# -----------------------------------------------------------------------------
run "nvidia-smi -L (device list)" nvidia-smi -L
run "nvidia-smi (driver/CUDA version, current utilization/memory)" nvidia-smi
run "nvidia-smi -q (full per-GPU query: memory, PCIe, clocks, power, ECC)" nvidia-smi -q
run "nvidia-smi topo -m (interconnect matrix + CPU/NUMA affinity)" nvidia-smi topo -m

# -----------------------------------------------------------------------------
section "GPU device properties via CUDA runtime (compute capability, SM count, memory bus, theoretical bandwidth)"
# -----------------------------------------------------------------------------
cat > /tmp/node_arch_probe.cu <<'EOF'
#include <cstdio>
#include <cuda_runtime.h>
int main() {
    int ngpu = 0;
    cudaError_t err = cudaGetDeviceCount(&ngpu);
    if (err != cudaSuccess) {
        printf("cudaGetDeviceCount FAILED: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaGetDeviceCount() = %d\n\n", ngpu);
    for (int i = 0; i < ngpu; i++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, i);
        double peak_bw_gbs = 2.0 * p.memoryClockRate * 1e3 * (p.memoryBusWidth / 8) / 1e9;
        printf("GPU %d: %s\n", i, p.name);
        printf("  compute capability      : %d.%d\n", p.major, p.minor);
        printf("  multiProcessorCount (SM): %d\n", p.multiProcessorCount);
        printf("  totalGlobalMem           : %.2f GiB\n", p.totalGlobalMem / 1073741824.0);
        printf("  memoryBusWidth           : %d bits\n", p.memoryBusWidth);
        printf("  memoryClockRate          : %.0f MHz\n", p.memoryClockRate / 1000.0);
        printf("  theoretical peak mem BW  : %.1f GB/s\n", peak_bw_gbs);
        printf("  l2CacheSize              : %d KiB\n", p.l2CacheSize / 1024);
        printf("  sharedMemPerBlock        : %zu KiB\n", p.sharedMemPerBlock / 1024);
        printf("  sharedMemPerMultiprocessor: %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("  regsPerMultiprocessor    : %d\n", p.regsPerMultiprocessor);
        printf("  warpSize                 : %d\n", p.warpSize);
        printf("  maxThreadsPerBlock       : %d\n", p.maxThreadsPerBlock);
        printf("  maxThreadsPerMultiProc   : %d\n", p.maxThreadsPerMultiProcessor);
        printf("  clockRate (SM)           : %.0f MHz\n", p.clockRate / 1000.0);
        printf("  pciBusID:pciDeviceID     : %02x:%02x.0\n", p.pciBusID, p.pciDeviceID);
        printf("  pciDomainID              : %d\n", p.pciDomainID);
        printf("  ECCEnabled               : %d\n", p.ECCEnabled);
        printf("  unifiedAddressing        : %d\n", p.unifiedAddressing);
        printf("  concurrentKernels        : %d\n", p.concurrentKernels);
        printf("  asyncEngineCount         : %d\n", p.asyncEngineCount);
        printf("\n");
    }
    return 0;
}
EOF
if nvcc -o /tmp/node_arch_probe /tmp/node_arch_probe.cu 2>&1; then
    /tmp/node_arch_probe
else
    echo "(nvcc compile failed, skipping CUDA device-properties probe)"
fi
rm -f /tmp/node_arch_probe /tmp/node_arch_probe.cu

# -----------------------------------------------------------------------------
section "PCIe topology"
# -----------------------------------------------------------------------------
run "lspci -tv (bus tree)"  lspci -tv
run "lspci NVIDIA devices, verbose (LnkCap/LnkSta = PCIe gen/width capable vs. actually negotiated)" \
    bash -c "lspci -d 10de: -vvv 2>/dev/null | grep -iE '^[0-9a-f]|LnkCap|LnkSta|ACSCtl' || echo '(no output -- may need elevated privileges for full link/ACS details)'"

# -----------------------------------------------------------------------------
section "Network / interconnect fabric"
# -----------------------------------------------------------------------------
run "ip -brief addr"        ip -brief addr
run "ip -brief link"        ip -brief link
run "InfiniBand devices (ibstat)" ibstat
run "InfiniBand devices (ibv_devinfo)" ibv_devinfo
run "RDMA links (rdma link)" rdma link
echo "NOTE: this project's jobs request --nodes=1 (single node, edu01, 4 local"
echo "GPUs) -- if no IB/RDMA devices are listed above, no multi-node fabric is"
echo "exercised by any driver in this project; all communication measured"
echo "throughout (MPI/NCCL) happens over the intra-node PCIe topology profiled"
echo "above, not a cluster interconnect."
echo ""

# -----------------------------------------------------------------------------
section "Software stack versions"
# -----------------------------------------------------------------------------
run "nvcc --version"        nvcc --version
run "mpirun --version"      mpirun --version
run "ompi_info -v ompi full (build config summary)" bash -c "ompi_info | head -30"
run "gcc --version"          gcc --version
run "cmake --version"        cmake --version
run "NCCL headers/version"   bash -c "find /opt/shares/NVHPC -iname 'nccl.h' 2>/dev/null | head -3 | xargs -I{} sh -c 'echo {}; grep -E \"NCCL_(MAJOR|MINOR|PATCH)\" {}'"
run "loaded environment modules" bash -c "module list 2>&1"

# -----------------------------------------------------------------------------
section "SLURM view of this node"
# -----------------------------------------------------------------------------
run "scontrol show node edu01" scontrol show node edu01
run "sinfo -N -l -n edu01"      sinfo -N -l -n edu01

# -----------------------------------------------------------------------------
section "Filesystem (matrix storage capacity)"
# -----------------------------------------------------------------------------
run "df -h on repo root and /tmp" bash -c "df -h . /tmp"

echo ""
echo "=== Job done ==="
