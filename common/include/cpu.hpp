#pragma once
#include <vector>
#include "matrix.hpp"

// Sequential CPU SpMV, used as the correctness reference every driver
// checks its distributed result against.
void spmv_cpu(int nrows,
              const CSRHost &csr,
              const std::vector<float> &x,
              std::vector<float> &y);

bool almost_equal(float a, float b, float abs_tol = 1e-2f, float rel_tol = 1e-2f);

// Prints at most 10 individual mismatches, then a one-line summary
// ("OK!" or total mismatch count).
void check_correctness(int nrows,
                       const std::vector<float> &h_y_cpu,
                       const std::vector<float> &h_y_gpu);
