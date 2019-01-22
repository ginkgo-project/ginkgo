/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2019

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

#ifndef GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_


#include <algorithm>
#include <numeric>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include <iostream>


namespace gko {
namespace test {
namespace detail {


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex<ValueType>(), ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex<ValueType>(), ValueType>::type
get_rand_value(Distribution &&dist, Generator &&gen)
{
    return ValueType(dist(gen), dist(gen));
}


}  // namespace detail


/**
 * Generates a random matrix.
 *
 * @tparam MatrixType  type of matrix to generate (matrix::Dense must implement
 *                     the interface `ConvertibleTo<MatrixType>`)
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num_rows  number of rows
 * @param num_cols  number of colums
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 */
template <typename MatrixType = matrix::Dense<>, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_matrix(
    size_type num_rows, size_type num_cols, NonzeroDistribution &&nonzero_dist,
    ValueDistribution &&value_dist, Engine &&engine,
    std::shared_ptr<const Executor> exec, MatrixArgs &&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;

    matrix_data<value_type, index_type> data{gko::dim<2>{num_rows, num_cols},
                                             {}};

    std::vector<size_type> col_idx(num_cols);
    iota(begin(col_idx), end(col_idx), size_type(0));

    for (size_type row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        auto nnz_in_row = static_cast<size_type>(nonzero_dist(engine));
        nnz_in_row = std::max(size_type(0), std::min(nnz_in_row, num_cols));
        // select a subset of `nnz_in_row` column indexes, and fill these
        // locations with random values
        shuffle(begin(col_idx), end(col_idx), engine);
        for_each(
            begin(col_idx), begin(col_idx) + nnz_in_row, [&](size_type col) {
                data.nonzeros.emplace_back(
                    row, col,
                    detail::get_rand_value<value_type>(value_dist, engine));
            });
    }

    data.ensure_row_major_order();

    // convert to the correct matrix type
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(data);
    return result;
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_
