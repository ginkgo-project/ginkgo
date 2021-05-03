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

#ifndef GKO_CORE_TEST_UTILS_BATCH_HPP_
#define GKO_CORE_TEST_UTILS_BATCH_HPP_


#include <ginkgo/core/base/matrix_data.hpp>


namespace gko {
namespace test {


template <typename MatrixType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_uniform_batch_random_matrix(
    const size_type batch_size, const size_type num_rows,
    const size_type num_cols, NonzeroDistribution &&nonzero_dist,
    ValueDistribution &&value_dist, Engine &&engine,
    std::shared_ptr<const Executor> exec, MatrixArgs &&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    using std::begin;
    using std::end;

    std::vector<size_type> col_idx(num_cols);
    std::iota(begin(col_idx), end(col_idx), size_type(0));
    std::vector<size_type> nnz_in_row(num_rows);

    // generate sparsity pattern
    for (size_type row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        const auto nnz_row = static_cast<size_type>(nonzero_dist(engine));
        nnz_in_row[row] = std::max(size_type(0), std::min(nnz_row, num_cols));
        std::shuffle(begin(col_idx), end(col_idx), engine);
    }

    std::vector<matrix_data<value_type, index_type>> batchmtx;
    batchmtx.reserve(batch_size);

    for (size_t ibatch = 0; ibatch < batch_size; ibatch++) {
        matrix_data<value_type, index_type> data{
            gko::dim<2>{num_rows, num_cols}, {}};
        for (size_type row = 0; row < num_rows; ++row) {
            std::for_each(begin(col_idx), begin(col_idx) + nnz_in_row[row],
                          [&](size_type col) {
                              data.nonzeros.emplace_back(
                                  row, col,
                                  gko::detail::get_rand_value<value_type>(
                                      value_dist, engine));
                          });
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


#endif  // GKO_CORE_TEST_UTILS_BATCH_HPP_
