# Node architecture profile

One-shot diagnostic that collects every hardware/software fact about the
compute node this project's jobs run on, in one place.

---

## What it collects

| Category | Detail |
|---|---|
| **CPU** | model, socket/core/thread count, NUMA layout, cache sizes |
| **Memory** | total RAM, per-NUMA-node memory |
| **GPU** | model, per-GPU memory, compute capability, SM count, memory bus width/theoretical bandwidth, PCIe generation/width both *capable* and *currently negotiated* |
| **GPU interconnect topology** | NUMA/CPU affinity and PCIe ACS status (`nvidia-smi topo -m`) |
| **Network fabric** | confirms whether any multi-node interconnect/IB fabric is actually exercised |
| **Software stack** | CUDA/driver, Open MPI, NCCL, GCC, CMake versions actually in use |
| **SLURM's own view of the node** | partition membership, allocatable resources, node-level features/gres config |

---

## Running

```bash
sbatch node_architecture/sbatch/node_architecture_check.sh
```

No arguments. The script uses `set -uo pipefail` (deliberately without
`-e`) since most of the diagnostic commands it runs are allowed to fail or
be absent.

---

## Output

Plain-text diagnostic dump, not `RESULT` lines and not a CSV — meant to be
read directly, not parsed by an analysis script:

```
node_architecture/outputs/node_arch-<jobid>.out
node_architecture/outputs/node_arch-<jobid>.err
```
