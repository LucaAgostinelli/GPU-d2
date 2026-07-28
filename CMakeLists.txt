cmake_minimum_required(VERSION 3.20)

project(d2_final LANGUAGES C CXX CUDA)

# Language standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

set(CMAKE_CUDA_ARCHITECTURES 80)   # Ampere (A30, edu cluster)

# Output folders
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

find_package(MPI REQUIRED)

# NCCL is bundled inside the NVHPC module tree, not a standalone module --
# located by explicit path instead of `module load`-ing NVHPC wholesale.
set(NCCL_ROOT "/opt/shares/NVHPC/nvhpc_24.7_cuda_12.5/Linux_x86_64/24.7/comm_libs/nccl"
    CACHE PATH "Root of the NVHPC-bundled NCCL install (headers in include/, lib in lib/)")
find_path(NCCL_INCLUDE_DIR nccl.h HINTS ${NCCL_ROOT}/include)
find_library(NCCL_LIBRARY NAMES nccl HINTS ${NCCL_ROOT}/lib)

set(HAVE_NCCL FALSE)
if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    set(HAVE_NCCL TRUE)
else()
    message(WARNING
        "NCCL not found at NCCL_ROOT=${NCCL_ROOT} -- skipping the 4 NCCL-based "
        "targets (spmv_ghost_nccl_cusparse, spmv_ghost_nccl_acc, "
        "spmv_2d_nccl_cusparse, spmv_2d_nccl_acc). Pass -DNCCL_ROOT=<path> "
        "to cmake if it lives elsewhere.")
endif()

# Shared sources
set(COMMON_SOURCES
    common/matrix.cpp
    common/matrix_device.cu
    common/cpu.cpp
    common/random.cpp
)

# Shared setup for every GPU/MPI driver target.
function(configure_spmv_target name)
    target_include_directories(${name} PRIVATE
        common/include kernels src/include
        ${MPI_CXX_INCLUDE_DIRS}
    )
    target_link_libraries(${name} PRIVATE MPI::MPI_CXX m)
    target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
endfunction()

function(configure_nccl_target name)
    configure_spmv_target(${name})
    target_include_directories(${name} PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(${name} PRIVATE ${NCCL_LIBRARY})
    set_target_properties(${name} PROPERTIES BUILD_RPATH ${NCCL_ROOT}/lib)
endfunction()

# 1D: full broadcast reference (cuSPARSE only)
add_executable(spmv_bcast
    drivers/spmv_bcast.cu
    ${COMMON_SOURCES}
    kernels/cusparse.cu
    src/distribute_matrix.cpp
)
configure_spmv_target(spmv_bcast)
target_link_libraries(spmv_bcast PRIVATE cusparse)

# ACC-kernel twin of spmv_bcast, for kernel-matched comparisons.
add_executable(spmv_bcast_acc
    drivers/spmv_bcast_acc.cu
    ${COMMON_SOURCES}
    kernels/acc.cu
    src/distribute_matrix.cpp
)
configure_spmv_target(spmv_bcast_acc)

# 1D: selective ghost exchange, MPI point-to-point
add_executable(spmv_ghost_cusparse
    drivers/1d/spmv_ghost_cusparse.cu
    ${COMMON_SOURCES}
    kernels/cusparse.cu
    src/distribute_matrix.cpp
    src/ghost_exchange.cpp
)
configure_spmv_target(spmv_ghost_cusparse)
target_link_libraries(spmv_ghost_cusparse PRIVATE cusparse)

add_executable(spmv_ghost_acc
    drivers/1d/spmv_ghost_acc.cu
    ${COMMON_SOURCES}
    kernels/acc.cu
    src/distribute_matrix.cpp
    src/ghost_exchange.cpp
)
configure_spmv_target(spmv_ghost_acc)

# 2D checkerboard, MPI Bcast/Reduce
add_executable(spmv_2d_cusparse
    drivers/2d/spmv_2d_cusparse.cu
    ${COMMON_SOURCES}
    kernels/cusparse.cu
    src/distribute_matrix.cpp
    src/distribute_matrix_2d.cpp
    src/proc_grid.cpp
)
configure_spmv_target(spmv_2d_cusparse)
target_link_libraries(spmv_2d_cusparse PRIVATE cusparse)

add_executable(spmv_2d_acc
    drivers/2d/spmv_2d_acc.cu
    ${COMMON_SOURCES}
    kernels/acc.cu
    src/distribute_matrix.cpp
    src/distribute_matrix_2d.cpp
    src/proc_grid.cpp
)
configure_spmv_target(spmv_2d_acc)

# NCCL-based targets (skipped, with the warning above, if NCCL isn't found)
if(HAVE_NCCL)
    add_executable(spmv_ghost_nccl_cusparse
        drivers/1d/spmv_ghost_nccl_cusparse.cu
        ${COMMON_SOURCES}
        kernels/cusparse.cu
        src/distribute_matrix.cpp
        src/ghost_exchange.cpp
    )
    configure_nccl_target(spmv_ghost_nccl_cusparse)
    target_link_libraries(spmv_ghost_nccl_cusparse PRIVATE cusparse)

    add_executable(spmv_ghost_nccl_acc
        drivers/1d/spmv_ghost_nccl_acc.cu
        ${COMMON_SOURCES}
        kernels/acc.cu
        src/distribute_matrix.cpp
        src/ghost_exchange.cpp
    )
    configure_nccl_target(spmv_ghost_nccl_acc)

    add_executable(spmv_2d_nccl_cusparse
        drivers/2d/spmv_2d_nccl_cusparse.cu
        ${COMMON_SOURCES}
        kernels/cusparse.cu
        src/distribute_matrix.cpp
        src/distribute_matrix_2d.cpp
        src/proc_grid.cpp
    )
    configure_nccl_target(spmv_2d_nccl_cusparse)
    target_link_libraries(spmv_2d_nccl_cusparse PRIVATE cusparse)

    add_executable(spmv_2d_nccl_acc
        drivers/2d/spmv_2d_nccl_acc.cu
        ${COMMON_SOURCES}
        kernels/acc.cu
        src/distribute_matrix.cpp
        src/distribute_matrix_2d.cpp
        src/proc_grid.cpp
    )
    configure_nccl_target(spmv_2d_nccl_acc)

    # Bonus: RCM reordering + NNZ-balanced block partitioning, NCCL transport, ACC kernel.
    add_executable(spmv_rcm_nccl_acc
        drivers/partitioning/spmv_rcm_nccl_acc.cu
        ${COMMON_SOURCES}
        kernels/acc.cu
        src/rcm_partition.cpp
        src/distribute_matrix_partitioned.cpp
        src/ghost_exchange_mapped.cpp
    )
    configure_nccl_target(spmv_rcm_nccl_acc)

    # Bonus: streaming greedy LDG edge-cut partitioning, NCCL transport, ACC kernel.
    add_executable(spmv_fennel_nccl_acc
        drivers/partitioning/spmv_fennel_nccl_acc.cu
        ${COMMON_SOURCES}
        kernels/acc.cu
        src/fennel_partition.cpp
        src/distribute_matrix_partitioned.cpp
        src/ghost_exchange_mapped.cpp
    )
    configure_nccl_target(spmv_fennel_nccl_acc)

    # Kernel-isolation twins: same partitioning + NCCL transport, cuSPARSE instead of ACC.
    add_executable(spmv_rcm_nccl_cusparse
        drivers/partitioning/spmv_rcm_nccl_cusparse.cu
        ${COMMON_SOURCES}
        kernels/cusparse.cu
        src/rcm_partition.cpp
        src/distribute_matrix_partitioned.cpp
        src/ghost_exchange_mapped.cpp
    )
    configure_nccl_target(spmv_rcm_nccl_cusparse)
    target_link_libraries(spmv_rcm_nccl_cusparse PRIVATE cusparse)

    add_executable(spmv_fennel_nccl_cusparse
        drivers/partitioning/spmv_fennel_nccl_cusparse.cu
        ${COMMON_SOURCES}
        kernels/cusparse.cu
        src/fennel_partition.cpp
        src/distribute_matrix_partitioned.cpp
        src/ghost_exchange_mapped.cpp
    )
    configure_nccl_target(spmv_fennel_nccl_cusparse)
    target_link_libraries(spmv_fennel_nccl_cusparse PRIVATE cusparse)

    # Bonus: classical 1D-Block partitioning (contiguous, no NNZ balancing), NCCL transport, ACC kernel.
    add_executable(spmv_block_nccl_acc
        drivers/partitioning/spmv_block_nccl_acc.cu
        ${COMMON_SOURCES}
        kernels/acc.cu
        src/block_partition.cpp
        src/distribute_matrix_partitioned.cpp
        src/ghost_exchange_mapped.cpp
    )
    configure_nccl_target(spmv_block_nccl_acc)

    # Kernel-isolation twin: cuSPARSE instead of ACC.
    add_executable(spmv_block_nccl_cusparse
        drivers/partitioning/spmv_block_nccl_cusparse.cu
        ${COMMON_SOURCES}
        kernels/cusparse.cu
        src/block_partition.cpp
        src/distribute_matrix_partitioned.cpp
        src/ghost_exchange_mapped.cpp
    )
    configure_nccl_target(spmv_block_nccl_cusparse)
    target_link_libraries(spmv_block_nccl_cusparse PRIVATE cusparse)
endif()

# Distributed read-phase benchmark (bonus): pure MPI + host C++, no CUDA/GPU/NCCL.
add_executable(bench_mtx_read
    drivers/bench_mtx_read.cpp
    common/matrix.cpp
    common/cpu.cpp
    common/random.cpp
    src/distribute_matrix.cpp
    src/distribute_matrix_chunked.cpp
)
target_include_directories(bench_mtx_read PRIVATE
    common/include src/include
    ${MPI_CXX_INCLUDE_DIRS}
)
target_link_libraries(bench_mtx_read PRIVATE MPI::MPI_CXX m)

# Serial CPU baseline -- plain C++, no CUDA or MPI at all.
# mmio.c is compiled as C++ (mmio.h has no extern "C" guards).
set_source_files_properties(common/mmio.c PROPERTIES LANGUAGE CXX)

add_executable(spmv_cpu_baseline
    drivers/spmv_cpu_baseline.cpp
    common/mmio.c
)
target_include_directories(spmv_cpu_baseline PRIVATE common/include)
