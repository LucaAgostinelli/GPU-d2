#include "distribute_matrix_chunked.hpp"
#include "matrix.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstring>
#include <cstdlib>
#include <stdexcept>

namespace
{

struct MtxHeader
{
    int nrows = 0, ncols = 0;
    bool symmetric = false, skew = false, pattern = false;
    std::streamoff data_start_offset = 0; // first byte of the first data line
};

// Every rank parses the header itself -- a plain ifstream reading a
// handful of lines, no MPI call of any kind. Re-parsing it on every rank
// instead of having one rank Bcast it removes a collective from the read
// path entirely; the header is tiny, so the redundant work costs nothing.
MtxHeader parse_header(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + path);

    std::string line;
    std::getline(file, line);

    bool skew = (line.find("skew-symmetric") != std::string::npos);
    bool symmetric = skew || (line.find("symmetric") != std::string::npos);
    bool pattern = (line.find("pattern") != std::string::npos);

    while (std::getline(file, line))
    {
        if (line[0] != '%')
            break;
    }

    std::istringstream ss(line);
    int nrows = 0, ncols = 0;
    long long nnz = 0;
    ss >> nrows >> ncols >> nnz;

    MtxHeader h;
    h.nrows = nrows;
    h.ncols = ncols;
    h.symmetric = symmetric;
    h.skew = skew;
    h.pattern = pattern;
    h.data_start_offset = file.tellg();
    return h;
}

// Byte offset of the start of the next full line at-or-after `from` (skips
// forward past any partial line straddling `from`). Two ranks calling this
// on the same (file content, offset) always agree -- it's a pure function
// of the bytes on disk -- which is what lets neighboring shards line up
// exactly with no data duplicated or dropped and no inter-rank handshake.
std::streamoff next_line_start(std::ifstream &file, std::streamoff from, std::streamoff file_size)
{
    const std::streamsize PROBE = 1 << 16;
    std::vector<char> buf((size_t)PROBE);
    std::streamoff pos = from;
    while (pos < file_size)
    {
        file.seekg(pos);
        std::streamsize want = std::min(PROBE, (std::streamsize)(file_size - pos));
        file.read(buf.data(), want);
        std::streamsize got = file.gcount();
        if (got == 0)
            break;
        for (std::streamsize k = 0; k < got; k++)
        {
            if (buf[(size_t)k] == '\n')
                return pos + k + 1;
        }
        pos += got;
    }
    return file_size;
}

// Parses this rank's shard (buf[0..n), null-terminated at buf[n]) into
// global-space COO entries directly off the raw buffer -- strtol/strtof
// skip leading whitespace and advance a pointer past whatever they parsed
// on their own, so no line is ever copied into a std::string or tokenized
// through an istringstream.
void parse_buffer(const char *buf, size_t n, const MtxHeader &h, std::vector<COO> &out)
{
    const char *p = buf;
    const char *end = buf + n;
    while (p < end)
    {
        const char *nl = (const char *)memchr(p, '\n', (size_t)(end - p));
        const char *line_end = nl ? nl : end;

        const char *cursor = p;
        while (cursor < line_end && (*cursor == ' ' || *cursor == '\t' || *cursor == '\r'))
            cursor++;

        if (cursor < line_end && *cursor != '%')
        {
            char *after = nullptr;
            long row = std::strtol(cursor, &after, 10);
            long col = std::strtol(after, &after, 10);
            float val = 1.0f;
            if (!h.pattern)
                val = std::strtof(after, &after);

            row--;
            col--;
            out.push_back({(int)row, (int)col, val});
            if (h.symmetric && row != col)
                out.push_back({(int)col, (int)row, h.skew ? -val : val});
        }

        p = nl ? nl + 1 : end;
    }
}

} // namespace

LocalCSR read_and_distribute_chunked(const std::string &path, MPI_Comm comm,
                                     ChunkedReadMetrics *metrics)
{
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);

    double t_io0 = MPI_Wtime();

    // 1) Every rank opens the file and parses the header itself:
    //    zero MPI calls anywhere in this step
    MtxHeader h = parse_header(path);

    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error("Cannot open file: " + path);
    file.seekg(0, std::ios::end);
    std::streamoff file_size = file.tellg();

    // 2) Byte-range partition of the data region, aligned to lines,
    //    computed independently by every rank -- no communication
    std::streamoff data_bytes = file_size - h.data_start_offset;
    std::streamoff chunk = data_bytes / P;
    std::streamoff raw_start = h.data_start_offset + (std::streamoff)rank * chunk;
    std::streamoff raw_end = (rank == P - 1) ? file_size
                                             : h.data_start_offset + (std::streamoff)(rank + 1) * chunk;

    std::streamoff real_start = (rank == 0) ? h.data_start_offset
                                            : next_line_start(file, raw_start, file_size);
    std::streamoff real_end = (rank == P - 1) ? file_size
                                              : next_line_start(file, raw_end, file_size);

    long long local_bytes = (long long)(real_end - real_start);
    if (local_bytes < 0)
        local_bytes = 0;

    // 3) Plain sequential read of this rank's own shard: no MPI
    //    collective I/O call anywhere, just ifstream::read. One extra
    //    byte is allocated and explicitly null-terminated so
    //    parse_buffer's strtol/strtof calls can never read past the
    //    shard on an unterminated final line. 
    std::vector<char> buf((size_t)local_bytes + 1);
    buf[(size_t)local_bytes] = '\0';
    if (local_bytes > 0)
    {
        file.seekg(real_start);
        file.read(buf.data(), (std::streamsize)local_bytes);
    }
    file.close();

    double t_io1 = MPI_Wtime();

    // 4) Parse this rank's shard into global-space COO
    std::vector<COO> local_coo;
    parse_buffer(buf.data(), (size_t)local_bytes, h, local_coo);
    buf.clear();
    buf.shrink_to_fit();

    double t_parse1 = MPI_Wtime();

    // 5) Redistribute by owner(row) = row % P (same rule
    //    distribute_matrix() uses), rewriting to local row index
    //    (row / P) before packing
    std::vector<std::vector<COO>> buckets(P);
    for (const COO &e : local_coo)
    {
        int owner = e.row % P;
        int local_row = e.row / P;
        buckets[owner].push_back({local_row, e.col, e.val});
    }
    local_coo.clear();
    local_coo.shrink_to_fit();

    std::vector<int> send_counts(P), recv_counts(P);
    for (int r = 0; r < P; r++)
        send_counts[r] = (int)buckets[r].size();
    MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, comm);

    std::vector<int> send_displs(P), recv_displs(P);
    int send_total = 0, recv_total = 0;
    for (int r = 0; r < P; r++)
    {
        send_displs[r] = send_total;
        send_total += send_counts[r];
        recv_displs[r] = recv_total;
        recv_total += recv_counts[r];
    }

    std::vector<int> send_row(send_total), send_col(send_total);
    std::vector<float> send_val(send_total);
    for (int r = 0; r < P; r++)
    {
        int off = send_displs[r];
        for (const COO &e : buckets[r])
        {
            send_row[off] = e.row;
            send_col[off] = e.col;
            send_val[off] = e.val;
            off++;
        }
    }
    buckets.clear();
    buckets.shrink_to_fit();

    std::vector<int> recv_row(recv_total), recv_col(recv_total);
    std::vector<float> recv_val(recv_total);
    MPI_Alltoallv(send_row.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_row.data(), recv_counts.data(), recv_displs.data(), MPI_INT, comm);
    MPI_Alltoallv(send_col.data(), send_counts.data(), send_displs.data(), MPI_INT,
                  recv_col.data(), recv_counts.data(), recv_displs.data(), MPI_INT, comm);
    MPI_Alltoallv(send_val.data(), send_counts.data(), send_displs.data(), MPI_FLOAT,
                  recv_val.data(), recv_counts.data(), recv_displs.data(), MPI_FLOAT, comm);

    std::vector<COO> my_coo(recv_total);
    for (int k = 0; k < recv_total; k++)
        my_coo[k] = {recv_row[k], recv_col[k], recv_val[k]};

    // 6) Local COO -> CSR (each rank converts its own shard)
    LocalCSR out;
    out.rank = rank;
    out.P = P;
    out.nrows_global = h.nrows;
    out.ncols_global = h.ncols;
    out.nrows_local = local_nrows(h.nrows, P, rank);

    CSRHost csr = coo_to_csr(my_coo, out.nrows_local, h.ncols);
    out.nnz_local = csr.nnz;
    out.row_ptr = std::move(csr.row_ptr);
    out.col_idx = std::move(csr.col_idx);
    out.values = std::move(csr.values);

    double t_redist1 = MPI_Wtime();

    if (metrics)
    {
        metrics->io_ms = (t_io1 - t_io0) * 1000.0;
        metrics->parse_ms = (t_parse1 - t_io1) * 1000.0;
        metrics->redistribute_ms = (t_redist1 - t_parse1) * 1000.0;
    }

    return out;
}
