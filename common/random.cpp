#include "random.hpp"
#include <random>

std::vector<float> generateRandomArray(int size)
{
    std::vector<float> arr(size);
    std::mt19937 gen(42); // fixed seed for reproducibility
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < size; i++)
        arr[i] = dist(gen);

    return arr;
}
