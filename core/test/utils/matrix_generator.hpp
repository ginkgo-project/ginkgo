/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_


#include <algorithm>
#include <iterator>
#include <numeric>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/value_generator.hpp"


namespace gko {
namespace test {


/**
 * Generates matrix data for a random matrix.
 *
 * @tparam ValueType  the type for matrix values
 * @tparam IndexType  the type for row and column indices
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 *
 * @return the generated matrix_data with entries according to the given
 *         dimensions and nonzero count and value distributions.
 */
template <typename ValueType, typename IndexType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine>
matrix_data<ValueType, IndexType> generate_random_matrix_data(
    size_type num_rows, size_type num_cols, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine)
{
    using std::begin;
    using std::end;

    matrix_data<ValueType, IndexType> data{gko::dim<2>{num_rows, num_cols}, {}};

    std::vector<size_type> col_idx(num_cols);
    std::iota(begin(col_idx), end(col_idx), size_type(0));

    for (size_type row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        auto nnz_in_row = static_cast<size_type>(nonzero_dist(engine));
        nnz_in_row = std::max(size_type(0), std::min(nnz_in_row, num_cols));
        // select a subset of `nnz_in_row` column indexes, and fill these
        // locations with random values
        std::shuffle(begin(col_idx), end(col_idx), engine);
        std::for_each(
            begin(col_idx), begin(col_idx) + nnz_in_row, [&](size_type col) {
                data.nonzeros.emplace_back(
                    row, col,
                    detail::get_rand_value<ValueType>(value_dist, engine));
            });
    }

    data.ensure_row_major_order();
    return data;
}


/**
 * Generates device matrix data for a random matrix.
 *
 * @see generate_random_matrix_data
 */
template <typename ValueType, typename IndexType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine>
gko::device_matrix_data<ValueType, IndexType>
generate_random_device_matrix_data(gko::size_type num_rows,
                                   gko::size_type num_cols,
                                   NonzeroDistribution&& nonzero_dist,
                                   ValueDistribution&& value_dist,
                                   Engine&& engine,
                                   std::shared_ptr<const gko::Executor> exec)
{
    auto md = gko::test::generate_random_matrix_data<ValueType, IndexType>(
        num_rows, num_cols, std::forward<NonzeroDistribution>(nonzero_dist),
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine));
    md.ensure_row_major_order();
    return gko::device_matrix_data<ValueType, IndexType>::create_from_host(exec,
                                                                           md);
}


/**
 * Generates a random matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 *
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * The other (template) parameters match generate_random_matrix_data.
 *
 * @return the unique pointer of MatrixType
 */
template <typename MatrixType = matrix::Dense<>, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_matrix(
    size_type num_rows, size_type num_cols, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;

    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(generate_random_matrix_data<value_type, index_type>(
        num_rows, num_cols, std::forward<NonzeroDistribution>(nonzero_dist),
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine)));
    return result;
}


/**
 * Generates a random triangular matrix.
 *
 * @tparam ValueType  the type for matrix values
 * @tparam IndexType  the type for row and column indices
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param ones_on_diagonal  `true` generates only ones on the diagonal,
 *                          `false` generates random values on the diagonal
 * @param lower_triangular  `true` generates a lower triangular matrix,
 *                          `false` an upper triangular matrix
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 *
 * @return the generated matrix_data with entries according to the given
 *         dimensions and nonzero count and value distributions.
 */
template <typename ValueType, typename IndexType, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine>
matrix_data<ValueType, IndexType> generate_random_triangular_matrix_data(
    size_type num_rows, size_type num_cols, bool ones_on_diagonal,
    bool lower_triangular, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine)
{
    using std::begin;
    using std::end;

    matrix_data<ValueType, IndexType> data{gko::dim<2>{num_rows, num_cols}, {}};
    ValueType one = 1.0;
    std::vector<size_type> col_idx(num_cols);
    std::iota(begin(col_idx), end(col_idx), size_type(0));

    for (size_type row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        auto nnz_in_row = static_cast<size_type>(nonzero_dist(engine));
        nnz_in_row = std::max(size_type(0), std::min(nnz_in_row, num_cols));
        // select a subset of `nnz_in_row` column indexes, and fill these
        // locations with random values
        std::shuffle(begin(col_idx), end(col_idx), engine);
        // add non-zeros
        bool has_diagonal{};
        for (size_type nz = 0; nz < nnz_in_row; ++nz) {
            auto col = col_idx[nz];
            // skip non-zeros outside triangle
            if ((col > row && lower_triangular) ||
                (col < row && !lower_triangular)) {
                continue;
            }

            // generate and store non-zero
            auto val = detail::get_rand_value<ValueType>(value_dist, engine);
            if (col == row) {
                has_diagonal = true;
                if (ones_on_diagonal) {
                    val = one;
                }
            }
            data.nonzeros.emplace_back(row, col, val);
        }

        // add diagonal if it hasn't been added yet
        if (!has_diagonal && row < num_cols) {
            auto val = ones_on_diagonal ? one
                                        : detail::get_rand_value<ValueType>(
                                              value_dist, engine);
            data.nonzeros.emplace_back(row, row, val);
        }
    }

    data.ensure_row_major_order();
    return data;
}


/**
 * Generates a random triangular matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 *
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * The other (template) parameters match generate_random_triangular_matrix_data.
 *
 * @return the unique pointer of MatrixType
 */
template <typename MatrixType = matrix::Dense<>, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_triangular_matrix(
    size_type num_rows, size_type num_cols, bool ones_on_diagonal,
    bool lower_triangular, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(generate_random_triangular_matrix_data<value_type, index_type>(
        num_rows, num_cols, ones_on_diagonal, lower_triangular,
        std::forward<NonzeroDistribution>(nonzero_dist),
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine)));
    return result;
}


/**
 * Generates a random lower triangular matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 * @tparam MatrixArgs  the arguments from the matrix to be forwarded.
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param ones_on_diagonal  `true` generates only ones on the diagonal,
 *                          `false` generates random values on the diagonal
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * @return the unique pointer of MatrixType
 */
template <typename MatrixType = matrix::Dense<>, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_lower_triangular_matrix(
    size_type num_rows, size_type num_cols, bool ones_on_diagonal,
    NonzeroDistribution&& nonzero_dist, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    return generate_random_triangular_matrix<MatrixType>(
        num_rows, num_cols, ones_on_diagonal, true, nonzero_dist, value_dist,
        engine, std::move(exec), std::forward<MatrixArgs>(args)...);
}


/**
 * Generates a random upper triangular matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 * @tparam MatrixArgs  the arguments from the matrix to be forwarded.
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param ones_on_diagonal  `true` generates only ones on the diagonal,
 *                          `false` generates random values on the diagonal
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * @return the unique pointer of MatrixType
 */
template <typename MatrixType = matrix::Dense<>, typename NonzeroDistribution,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_upper_triangular_matrix(
    size_type num_rows, size_type num_cols, bool ones_on_diagonal,
    NonzeroDistribution&& nonzero_dist, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    return generate_random_triangular_matrix<MatrixType>(
        num_rows, num_cols, ones_on_diagonal, false, nonzero_dist, value_dist,
        engine, std::move(exec), std::forward<MatrixArgs>(args)...);
}


/**
 * Generates a random square band matrix.
 *
 * @tparam ValueType  the type for matrix values
 * @tparam IndexType  the type for row and column indices
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param size  number of rows and columns
 * @param lower_bandwidth number of nonzeros in each row left of the main
 * diagonal
 * @param upper_bandwidth number of nonzeros in each row right of the main
 * diagonal
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 *
 * @return the generated matrix_data with entries according to the given
 *         dimensions and nonzero count and value distributions.
 */
template <typename ValueType, typename IndexType, typename ValueDistribution,
          typename Engine, typename... MatrixArgs>
matrix_data<ValueType, IndexType> generate_random_band_matrix_data(
    size_type size, size_type lower_bandwidth, size_type upper_bandwidth,
    ValueDistribution&& value_dist, Engine&& engine)
{
    matrix_data<ValueType, IndexType> data{gko::dim<2>{size, size}, {}};
    for (size_type row = 0; row < size; ++row) {
        for (size_type col = row < lower_bandwidth ? 0 : row - lower_bandwidth;
             col <= std::min(row + upper_bandwidth, size - 1); col++) {
            auto val = detail::get_rand_value<ValueType>(value_dist, engine);
            data.nonzeros.emplace_back(row, col, val);
        }
    }
    return data;
}


/**
 * Generates a random banded matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 *
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * The other (template) parameters match generate_random_band_matrix_data.
 *
 * @return the unique pointer of MatrixType
 */
template <typename MatrixType = matrix::Dense<>, typename ValueDistribution,
          typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> generate_random_band_matrix(
    size_type size, size_type lower_bandwidth, size_type upper_bandwidth,
    ValueDistribution&& value_dist, Engine&& engine,
    std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(generate_random_band_matrix_data<value_type, index_type>(
        size, lower_bandwidth, upper_bandwidth,
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine)));
    return result;
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_
