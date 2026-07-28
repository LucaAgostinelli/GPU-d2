#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Used by every GPU/MPI driver and kernel module.
#define CUDA_CHECK(call)                                         \
    do                                                           \
    {                                                            \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess)                                  \
        {                                                        \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(1);                                             \
        }                                                        \
    } while (0)
