/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <complex>
#include <iostream>
#include <vector>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>


#ifdef GKO_TOOL_COMPLEX
using value_type = std::complex<double>;
#else
using value_type = double;
#endif


using matrix_data = gko::matrix_data<value_type, gko::int64>;


matrix_data make_lower_triangular(const matrix_data& data)
{
    matrix_data out(data.size);
    for (auto entry : data.nonzeros) {
        if (entry.column <= entry.row) {
            out.nonzeros.push_back(entry);
        }
    }
    return out;
}


matrix_data make_upper_triangular(const matrix_data& data)
{
    matrix_data out(data.size);
    for (auto entry : data.nonzeros) {
        if (entry.column >= entry.row) {
            out.nonzeros.push_back(entry);
        }
    }
    return out;
}


matrix_data make_remove_diagonal(matrix_data data)
{
    data.nonzeros.erase(
        std::remove_if(data.nonzeros.begin(), data.nonzeros.end(),
                       [](auto entry) { return entry.row == entry.column; }),
        data.nonzeros.end());
    return data;
}


matrix_data make_unit_diagonal(matrix_data data)
{
    data = make_remove_diagonal(data);
    auto num_diags = std::min(data.size[0], data.size[1]);
    for (gko::int64 i = 0; i < num_diags; i++) {
        data.nonzeros.emplace_back(i, i, 1.0);
    }
    data.ensure_row_major_order();
    return data;
}


matrix_data make_remove_zeros(matrix_data data)
{
    data.nonzeros.erase(
        std::remove_if(data.nonzeros.begin(), data.nonzeros.end(),
                       [](auto entry) { return entry.value == value_type{}; }),
        data.nonzeros.end());
    return data;
}


template <typename Op>
matrix_data make_symmetric_generic(const matrix_data& data, Op op)
{
    matrix_data out(data.size);
    // compute A + op(A^T)
    for (auto entry : data.nonzeros) {
        out.nonzeros.emplace_back(entry);
        out.nonzeros.emplace_back(entry.column, entry.row, op(entry.value));
    }
    out.ensure_row_major_order();
    // combine matching nonzeros
    matrix_data out_compressed(data.size);
    auto it = out.nonzeros.begin();
    while (it != out.nonzeros.end()) {
        auto entry = *it;
        it++;
        for (; it != out.nonzeros.end() && it->row == entry.row &&
               it->column == entry.column;
             ++it) {
            entry.value += it->value;
        }
        // store sum of entries at (row, column) divided by 2
        out_compressed.nonzeros.emplace_back(entry.row, entry.column,
                                             entry.value / 2.0);
    }
    return out_compressed;
}

matrix_data make_diag_dominant(matrix_data data, double scale = 1.01)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(data.size);
    std::vector<double> norms(data.size[0]);
    std::vector<gko::int64> diag_positions(data.size[0], -1);
    gko::int64 i{};
    for (auto entry : data.nonzeros) {
        if (entry.row == entry.column) {
            diag_positions[entry.row] = i;
        } else {
            norms[entry.row] += gko::abs(entry.value);
        }
        i++;
    }
    for (gko::int64 i = 0; i < data.size[0]; i++) {
        if (diag_positions[i] < 0) {
            data.nonzeros.emplace_back(i, i, norms[i] * scale);
        } else {
            auto& diag_value = data.nonzeros[diag_positions[i]].value;
            const auto diag_magnitude = gko::abs(diag_value);
            const auto offdiag_magnitude = norms[i];
            if (diag_magnitude < offdiag_magnitude * scale) {
                const auto scaled_value =
                    diag_value * (offdiag_magnitude * scale / diag_magnitude);
                if (gko::is_finite(scaled_value)) {
                    diag_value = scaled_value;
                } else {
                    diag_value = offdiag_magnitude * scale;
                }
            }
        }
    }
    data.ensure_row_major_order();
    return data;
}


int main(int argc, char** argv)
{
    if (argc == 1) {
        std::cerr
            << "Usage: " << argv[0]
            << " [operation1] [operation2]\nApplies the given operations "
               "to the input matrix read from stdin\nand writes it to "
               "stdout.\nOperations are:\n"
               "  lower-triangular   removes nonzeros above the diagonal\n"
               "  upper-triangular   removes nonzeros below the diagonal\n"
               "  remove-diagonal    removes diagonal entries\n"
               "  remove-zeros       removes numerical zero entries\n"
               "  unit-diagonal      sets diagonal entries to zero\n"
               "  symmetric          computes (A + A^T)/2\n"
               "  skew-symmetric     computes (A - A^T)/2\n"
               "  hermitian          computes (A + A^H)/2\n"
               "  skew-hermitian     computes (A - A^H)/2\n"
               "  diagonal-dominant  scales diagonal entries so the\n"
               "                     matrix becomes diagonally dominant\n"
               "  spd                symmetric + diagonal-dominant\n"
               "  hpd                hermitian + diagonal-dominant"
            << std::endl;
        return 1;
    }

    auto data = gko::read_raw<value_type, gko::int64>(std::cin);
    data.ensure_row_major_order();
    for (int argi = 1; argi < argc; argi++) {
        std::string arg{argv[argi]};
        if (arg == "lower-triangular") {
            data = make_lower_triangular(data);
        } else if (arg == "upper-triangular") {
            data = make_upper_triangular(data);
        } else if (arg == "remove-diagonal") {
            data = make_remove_diagonal(data);
        } else if (arg == "remove-zeros") {
            data = make_remove_zeros(data);
        } else if (arg == "unit-diagonal") {
            data = make_unit_diagonal(data);
        } else if (arg == "symmetric") {
            data = make_symmetric_generic(data, [](auto v) { return v; });
        } else if (arg == "skew-symmetric") {
            data = make_symmetric_generic(data, [](auto v) { return -v; });
        } else if (arg == "hermitian") {
            data = make_symmetric_generic(data,
                                          [](auto v) { return gko::conj(v); });
        } else if (arg == "skew-hermitian") {
            data = make_symmetric_generic(data,
                                          [](auto v) { return -gko::conj(v); });
        } else if (arg == "diagonal-dominant") {
            data = make_diag_dominant(data);
        } else if (arg == "spd") {
            data = make_diag_dominant(
                make_symmetric_generic(data, [](auto v) { return v; }));
        } else if (arg == "hpd") {
            data = make_diag_dominant(make_symmetric_generic(
                data, [](auto v) { return gko::conj(v); }));
        } else {
            std::cerr << "Unknown operation " << arg << std::endl;
            return 1;
        }
    }
    gko::write_raw(std::cout, data, gko::layout_type::coordinate);
}
