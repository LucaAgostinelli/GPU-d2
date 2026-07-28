/**
 * Distributed-reading alternative to read_mtx() + distribute_matrix(): every
 * rank independently opens the .mtx file, parses the header itself (no
 * Bcast), computes its own line-aligned byte range of the data section, and
 * reads that range with a plain std::ifstream::read -- no MPI collective
 * I/O call anywhere in the read step. Parsed entries are then redistributed
 * by owner(i) = i % P via MPI_Alltoallv.
 *
 * Produces the exact same LocalCSR a caller would get from
 * distribute_matrix() given the same file. See
 * drivers/bench_mtx_read.cpp for the read-time comparison this
 * exists to support. Not wired into any SpMV driver.
 */
#pragma once

#include <string>
#include <mpi.h>

#include "distribute_matrix.hpp"

// Optional per-phase timing breakdown (filled in milliseconds, this rank's own view).
struct ChunkedReadMetrics
{
    double io_ms = 0.0;          // ifstream::read of this rank's own shard
    double parse_ms = 0.0;       // raw-buffer -> COO (strtol/strtof, no istringstream)
    double redistribute_ms = 0.0; // Alltoall(counts) + 3x Alltoallv + local COO->CSR
};

// Collective call: every rank in `comm` must call this together, and none
// is a "root" -- every rank opens the file and reads its own shard
// independently. `metrics` may be null.
LocalCSR read_and_distribute_chunked(const std::string &path, MPI_Comm comm,
                                     ChunkedReadMetrics *metrics = nullptr);
