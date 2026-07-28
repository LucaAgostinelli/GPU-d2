#pragma once
#include <vector>

// Fixed-seed uniform random vector in [-1, 1), used to generate the dense
// x vector for every driver
std::vector<float> generateRandomArray(int size);
