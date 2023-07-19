/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_TEST_UTILS_BATCH_HELPERS_HPP_
#define GKO_CORE_TEST_UTILS_BATCH_HELPERS_HPP_


#include <random>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/matrix_data.hpp>


#include "core/test/utils/assertions.hpp"


namespace gko {
namespace test {


/**
 * Converts a vector of unique pointers to a vector of shared pointers.
 */
template <typename T>
std::vector<std::shared_ptr<T>> share(std::vector<std::unique_ptr<T>>&& objs)
{
    std::vector<std::shared_ptr<T>> out;
    out.reserve(objs.size());
    for (auto& obj : objs) {
        out.push_back(std::move(obj));
    }
    return out;
}


/**
 * Generates a batch of random matrices of the specified type.
 */
template <typename MatrixType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_uniform_batch_random_matrix(
    const size_type batch_size, const size_type num_rows,
    const size_type num_cols, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    const bool with_all_diagonals, std::shared_ptr<const Executor> exec,
    MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;

    // generate sparsity pattern
    matrix_data<value_type, index_type> sdata{gko::dim<2>{num_rows, num_cols},
                                              {}};

    for (size_type row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        std::vector<size_type> col_idx(num_cols);
        std::iota(begin(col_idx), end(col_idx), size_type(0));
        const auto nnz_row = static_cast<size_type>(nonzero_dist(engine));
        size_type nnz_in_row =
            std::max(size_type(0), std::min(nnz_row, num_cols));
        std::shuffle(std::begin(col_idx), std::end(col_idx), engine);

        if (with_all_diagonals) {
            if (nnz_in_row == 0) {
                nnz_in_row = 1;
            }
            bool has_diagonal = false;
            for (size_type icol = 0; icol < nnz_in_row; icol++) {
                if (col_idx[icol] == row) {
                    has_diagonal = true;
                }
            }
            if (!has_diagonal) {
                col_idx[0] = row;
            }
        }

        std::for_each(
            std::begin(col_idx), std::begin(col_idx) + nnz_in_row,
            [&](size_type col) { sdata.nonzeros.emplace_back(row, col, 1.0); });
    }

    std::vector<matrix_data<value_type, index_type>> batchmtx;
    batchmtx.reserve(batch_size);

    for (size_t ibatch = 0; ibatch < batch_size; ibatch++) {
        matrix_data<value_type, index_type> data = sdata;
        for (size_type iz = 0; iz < data.nonzeros.size(); ++iz) {
            value_type valnz =
                gko::detail::get_rand_value<value_type>(value_dist, engine);
            if (data.nonzeros[iz].column == data.nonzeros[iz].row &&
                valnz == zero<value_type>()) {
                valnz = 1.0;
            }
            data.nonzeros[iz].value = valnz;
        }

        data.ensure_row_major_order();
        batchmtx.emplace_back(std::move(data));
    }

    // convert to the correct matrix type
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(batchmtx);
    return result;
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_BATCH_HELPERS_HPP_
