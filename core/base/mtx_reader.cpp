/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/base/mtx_reader.hpp"


#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>


extern "C" {
#include <mmio.h>
}  // extern "C"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"


namespace gko {


#define MMIO_CHECK(_call, _filename)                                    \
    do {                                                                \
        int errcode = _call;                                            \
        if (errcode != 0) {                                             \
            throw FILE_ERROR(_filename,                                 \
                             "MMIO error: " + std::to_string(errcode)); \
        }                                                               \
    } while (false)


namespace {


struct file_data {
    std::FILE *f;
    const std::string &filename;
};


template <typename T>
typename std::enable_if<!is_complex<T>(), T>::type combine(double rp, double ip)
{
    return static_cast<T>(rp);
}

template <typename T>
typename std::enable_if<is_complex<T>(), T>::type combine(double rp, double ip)
{
    using vt = typename T::value_type;
    return T(static_cast<vt>(rp), static_cast<vt>(ip));
}


template <typename ValueType, typename IndexType>
class read_modifier {
public:
    read_modifier(const MM_typecode &t, file_data &f) : t_(t), f_(f) {}

    int get_offset(int col)
    {
        if (mm_is_symmetric(t_) || mm_is_hermitian(t_)) {
            return col;
        } else if (mm_is_skew(t_)) {
            return col + 1;
        } else {
            return 0;
        }
    }

    void read_value_for(int row, int col,
                        matrix_data<ValueType, IndexType> &data)
    {
        using std::to_string;
        double rp = 0.0;
        double ip = 0.0;
        if (mm_is_complex(t_)) {
            if (fscanf(f_.f, "%lf %lf", &rp, &ip) != 2) {
                throw FILE_ERROR(
                    f_.filename,
                    "MMIO: unable to read value for matrix position " +
                        to_string(row) + ", " + to_string(col));
            }
        } else if (!mm_is_pattern(t_)) {
            if (fscanf(f_.f, "%lf", &rp) != 1) {
                throw FILE_ERROR(
                    f_.filename,
                    "MMIO: unable to read value for matrix position " +
                        to_string(row) + ", " + to_string(col));
            }
        } else {
            // pattern: the value is 1 (True)
            rp = 1.0;
        }
        data.nonzeros.emplace_back(row, col, combine<ValueType>(rp, ip));
        if (mm_is_symmetric(t_) && row != col) {
            data.nonzeros.emplace_back(col, row, combine<ValueType>(rp, ip));
        } else if (mm_is_skew(t_)) {
            data.nonzeros.emplace_back(col, row, combine<ValueType>(-rp, -ip));
        } else if (mm_is_hermitian(t_) && row != col) {
            data.nonzeros.emplace_back(col, row, combine<ValueType>(rp, -ip));
        }
    }

private:
    const MM_typecode &t_;
    file_data &f_;
};


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_sparse(file_data &f,
                                              const MM_typecode &t)
{
    using std::to_string;
    int m = 0;
    int n = 0;
    int k = 0;
    // TODO: this doesn't work for large matrices
    MMIO_CHECK(mm_read_mtx_crd_size(f.f, &m, &n, &k), f.filename);
    matrix_data<ValueType, IndexType> data;
    data.size.num_rows = m;
    data.size.num_cols = n;
    read_modifier<ValueType, IndexType> mod(t, f);
    for (int i = 0; i < k; ++i) {
        int row = 0;
        int col = 0;
        if (fscanf(f.f, "%d %d", &row, &col) != 2) {
            throw FILE_ERROR(f.filename, "MMIO: unable to read position " +
                                             to_string(i) +
                                             " for sparse matrix");
        }
        mod.read_value_for(row - 1, col - 1, data);
    }
    data.ensure_row_major_order();
    return data;
}


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_dense(file_data &f, const MM_typecode &t)
{
    int m = 0;
    int n = 0;
    // TODO: this doesn't work for large matrices
    MMIO_CHECK(mm_read_mtx_array_size(f.f, &m, &n), f.filename);
    matrix_data<ValueType, IndexType> data;
    data.size.num_rows = m;
    data.size.num_cols = n;
    read_modifier<ValueType, IndexType> mod(t, f);
    for (int col = 0; col < data.size.num_cols; ++col) {
        for (int row = mod.get_offset(col); row < data.size.num_rows; ++row) {
            mod.read_value_for(row, col, data);
        }
    }
    data.ensure_row_major_order();
    return data;
}


}  // namespace


template <typename ValueType, typename IndexType>
matrix_data<ValueType, IndexType> read_raw(const std::string &filename)
{
    std::unique_ptr<std::FILE, int (*)(std::FILE *)> f(
        std::fopen(filename.c_str(), "r"), std::fclose);
    if (f == nullptr) {
        throw FILE_ERROR(filename, std::strerror(errno));
    }
    MM_typecode t;
    MMIO_CHECK(mm_read_banner(f.get(), &t), filename);
    if (!is_complex<ValueType>() && mm_is_complex(t)) {
        throw FILE_ERROR(
            filename,
            "trying to read a complex matrix into a real storage type");
    }
    file_data fd{f.get(), filename};
    if (mm_is_sparse(t)) {
        return read_sparse<ValueType, IndexType>(fd, t);
    } else {
        return read_dense<ValueType, IndexType>(fd, t);
    }
}


#define DECLARE_READ_RAW(ValueType, IndexType) \
    matrix_data<ValueType, IndexType> read_raw(const std::string &filename)
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(DECLARE_READ_RAW);


}  // namespace gko
