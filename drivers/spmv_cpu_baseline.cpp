/**
 * Serial single-core CPU reference, adapted from the course-provided
 * baseline_multiGPU-SpMV/SpMV.cpp.
 *
 * Usage:
 *   ./spmv_cpu_baseline <matrix.mtx> [reps]     (reps defaults to 10)
 */
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>
#include <vector>
#include "mmio.h"

static std::string basename_of(const std::string &path)
{
    size_t pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

void coo_to_csr(int M, int nz, const int *I, const int *J, const double *val,
                 int **row_ptr_out, int **col_ind_out, float **values_out)
{
    int *row_ptr = (int *)calloc(M + 1, sizeof(int));
    int *col_ind = (int *)malloc(nz * sizeof(int));
    float *values = (float *)malloc(nz * sizeof(float));

    for (int i = 0; i < nz; i++)
        row_ptr[I[i] + 1]++;
    for (int i = 0; i < M; i++)
        row_ptr[i + 1] += row_ptr[i];

    std::vector<int> temp_row_ptr(row_ptr, row_ptr + M + 1);
    for (int i = 0; i < nz; i++)
    {
        int row = I[i];
        int dest = temp_row_ptr[row];
        col_ind[dest] = J[i];
        values[dest] = (float)val[i];
        temp_row_ptr[row]++;
    }

    *row_ptr_out = row_ptr;
    *col_ind_out = col_ind;
    *values_out = values;
}

void spmv_csr(int rows, const int *row_ptr, const int *col_ind,
              const float *values, const float *x, float *y)
{
    for (int i = 0; i < rows; i++)
    {
        float acc = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
            acc += values[j] * x[col_ind[j]];
        y[i] = acc;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <matrix-market-filename> [reps]\n", argv[0]);
        exit(1);
    }
    int reps = (argc >= 3) ? atoi(argv[2]) : 10;

    FILE *f = fopen(argv[1], "r");
    if (!f)
    {
        fprintf(stderr, "Cannot open %s\n", argv[1]);
        exit(1);
    }

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0)
    {
        fprintf(stderr, "Could not process Matrix Market banner.\n");
        exit(1);
    }
    if (mm_is_complex(matcode))
    {
        fprintf(stderr, "Complex matrices not supported.\n");
        exit(1);
    }

    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0)
        exit(1);

    bool symmetric = mm_is_symmetric(matcode);
    bool skew = mm_is_skew(matcode);
    bool mirror = symmetric || skew;
    bool pattern = mm_is_pattern(matcode);

    std::vector<int> I, J;
    std::vector<double> val;
    I.reserve((size_t)nz * (mirror ? 2 : 1));
    J.reserve((size_t)nz * (mirror ? 2 : 1));
    val.reserve((size_t)nz * (mirror ? 2 : 1));

    for (int k = 0; k < nz; k++)
    {
        int ii, jj;
        double vv;
        int nmatched;
        if (pattern)
        {
            nmatched = fscanf(f, "%d %d", &ii, &jj);
            vv = 1.0;
        }
        else
        {
            nmatched = fscanf(f, "%d %d %lg", &ii, &jj, &vv);
            if (nmatched < 3)
                vv = 1.0; // tolerate files that omit values despite the header
        }
        if (nmatched < 2)
        {
            fprintf(stderr, "Malformed entry at line %d, aborting.\n", k);
            exit(1);
        }
        ii--;
        jj--; // 0-based

        I.push_back(ii);
        J.push_back(jj);
        val.push_back(vv);
        if (mirror && ii != jj)
        {
            I.push_back(jj);
            J.push_back(ii);
            val.push_back(skew ? -vv : vv);
        }
    }
    fclose(f);

    int nnz_expanded = (int)I.size();

    int *row_ptr, *col_ind;
    float *values;
    coo_to_csr(M, nnz_expanded, I.data(), J.data(), val.data(), &row_ptr, &col_ind, &values);

    std::vector<float> x(N), y(M);
    unsigned seed = 12345u;
    for (int i = 0; i < N; i++)
    {
        seed = seed * 1103515245u + 12345u;
        x[i] = 2.0f * ((float)((seed >> 16) & 0x7fffu) / 32767.0f) - 1.0f;
    }

    // Warmup (not timed).
    spmv_csr(M, row_ptr, col_ind, values, x.data(), y.data());

    double total_ms = 0.0, min_ms = 1e18, max_ms = 0.0;
    for (int r = 0; r < reps; r++)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        spmv_csr(M, row_ptr, col_ind, values, x.data(), y.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }
    double avg_ms = total_ms / reps;
    double gflops = 2.0 * nnz_expanded / (avg_ms / 1e3) / 1e9;

    fprintf(stderr,
            "Loaded %s: %d x %d, nnz(file)=%d, nnz(expanded)=%d, symmetric=%d, skew=%d, pattern=%d\n",
            argv[1], M, N, nz, nnz_expanded, (int)symmetric, (int)skew, (int)pattern);

    printf("RESULT comm=none kernel=cpu_serial matrix=%s M=%d N=%d nnz_expanded=%d "
           "avg_ms=%.6f min_ms=%.6f max_ms=%.6f gflops=%.6f\n",
           basename_of(argv[1]).c_str(), M, N, nnz_expanded, avg_ms, min_ms, max_ms, gflops);

    free(row_ptr);
    free(col_ind);
    free(values);
    return 0;
}
