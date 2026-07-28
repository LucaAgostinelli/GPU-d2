#include "matrix.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

std::vector<COO> read_mtx(const std::string &filename,
                          int &nrows, int &ncols,
                          bool &symmetric)
{
    std::ifstream file(filename);
    if (!file)
        throw std::runtime_error("Cannot open file: " + filename);

    std::string line;
    std::getline(file, line);

    // "skew-symmetric" contains "symmetric" as a substring, so it must be
    // checked first -- otherwise a skew-symmetric banner would be
    // misclassified as plain symmetric and mirrored with the wrong sign.
    bool skew = (line.find("skew-symmetric") != std::string::npos);
    symmetric = skew || (line.find("symmetric") != std::string::npos);
    bool pattern = (line.find("pattern") != std::string::npos);

    while (std::getline(file, line))
    {
        if (line[0] != '%')
            break;
    }

    std::stringstream ss(line);
    int nnz;
    ss >> nrows >> ncols >> nnz;

    std::vector<COO> coo;
    coo.reserve(nnz * (symmetric ? 2 : 1));

    int i, j;
    float val;

    while (file >> i >> j)
    {
        // Pattern matrices store only (i, j) -- no value column. Reading a
        // value unconditionally here would eat the next line's row index
        // instead (`>>` skips newlines), desyncing every subsequent entry.
        if (pattern)
            val = 1.0f;
        else if (!(file >> val))
            val = 1.0f;

        // 0-based
        i--;
        j--;

        coo.push_back({i, j, val});

        if (symmetric && i != j)
            coo.push_back({j, i, skew ? -val : val});
    }

    return coo;
}

CSRHost coo_to_csr(std::vector<COO> &coo, int nrows, int ncols)
{
    CSRHost csr;
    csr.nrows = nrows;
    csr.ncols = ncols;
    csr.nnz = (int)coo.size();

    std::sort(coo.begin(), coo.end(), [](const COO &a, const COO &b)
              { return (a.row < b.row) || (a.row == b.row && a.col < b.col); });

    csr.row_ptr.assign(nrows + 1, 0);
    csr.col_idx.resize(csr.nnz);
    csr.values.resize(csr.nnz);

    for (const COO &e : coo)
        csr.row_ptr[e.row + 1]++;

    for (int i = 0; i < nrows; i++)
        csr.row_ptr[i + 1] += csr.row_ptr[i];

    std::vector<int> offset = csr.row_ptr;
    for (const COO &e : coo)
    {
        int pos = offset[e.row]++;
        csr.col_idx[pos] = e.col;
        csr.values[pos] = e.val;
    }

    return csr;
}
