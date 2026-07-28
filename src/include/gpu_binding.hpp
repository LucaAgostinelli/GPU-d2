#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>
#include "cuda_check.hpp"

// Binds this rank to GPU (rank % ngpu) and returns the peak memory
// bandwidth (GB/s) of the bound device, used to report %-of-peak achieved
// bandwidth alongside every result line. Aborts the whole job if no CUDA
// device is visible.
inline float bind_gpu_and_get_peak_bw(int rank)
{
    int ngpu = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpu));
    if (ngpu == 0)
    {
        if (rank == 0)
            fprintf(stderr, "No CUDA devices found. Aborting.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(rank % ngpu));

    int active_dev = -1;
    CUDA_CHECK(cudaGetDevice(&active_dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, active_dev));
    return 2.0f * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6f;
}
