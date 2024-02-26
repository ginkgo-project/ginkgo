// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_
#define GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_


#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/test/utils/value_generator.hpp"


namespace gko {
namespace test {


/**
 * Fills matrix data for a random matrix given a sparsity pattern
 *
 * @tparam ValueType  the type for matrix values
 * @tparam IndexType  the type for row and column indices
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param row_idxs  the row indices of the matrix
 * @param col_idxs  the column indices of the matrix
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 *
 * @return the generated matrix_data with entries according to the given
 *         dimensions and nonzero count and value distributions.
 */
template <typename ValueType, typename IndexType, typename ValueDistribution,
          typename Engine>
matrix_data<ValueType, IndexType> fill_random_matrix_data(
    size_type num_rows, size_type num_cols,
    const gko::array<IndexType>& row_indices,
    const gko::array<IndexType>& col_indices, ValueDistribution&& value_dist,
    Engine&& engine)
{
    matrix_data<ValueType, IndexType> data{gko::dim<2>{num_rows, num_cols}, {}};
    auto host_exec = row_indices.get_executor()->get_master();
    auto host_row_indices = make_temporary_clone(host_exec, &row_indices);
    auto host_col_indices = make_temporary_clone(host_exec, &col_indices);

    for (int nnz = 0; nnz < row_indices.get_size(); ++nnz) {
        data.nonzeros.emplace_back(
            host_row_indices->get_const_data()[nnz],
            host_col_indices->get_const_data()[nnz],
            detail::get_rand_value<ValueType>(value_dist, engine));
    }

    data.sort_row_major();
    return data;
}


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

    std::vector<bool> present_cols(num_cols);

    for (IndexType row = 0; row < num_rows; ++row) {
        // randomly generate number of nonzeros in this row
        const auto nnz_in_row = std::max(
            size_type(0),
            std::min(static_cast<size_type>(nonzero_dist(engine)), num_cols));
        std::uniform_int_distribution<IndexType> col_dist{
            0, static_cast<IndexType>(num_cols) - 1};
        if (nnz_in_row > num_cols / 2) {
            present_cols.assign(num_cols, true);
            // remove num_cols - nnz_in_row entries from present_cols
            size_type count = num_cols;
            while (count > nnz_in_row) {
                const auto new_col = col_dist(engine);
                if (present_cols[new_col]) {
                    present_cols[new_col] = false;
                    count--;
                }
            }
            for (IndexType col = 0; col < num_cols; col++) {
                if (present_cols[col]) {
                    data.nonzeros.emplace_back(
                        row, col,
                        detail::get_rand_value<ValueType>(value_dist, engine));
                }
            }
        } else {
            // add nnz_in_row entries to present_cols
            present_cols.assign(num_cols, false);
            size_type count = 0;
            while (count < nnz_in_row) {
                const auto col = col_dist(engine);
                if (!present_cols[col]) {
                    present_cols[col] = true;
                    count++;
                    data.nonzeros.emplace_back(
                        row, col,
                        detail::get_rand_value<ValueType>(value_dist, engine));
                }
            }
        }
    }

    data.sort_row_major();
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
    return gko::device_matrix_data<ValueType, IndexType>::create_from_host(exec,
                                                                           md);
}


/**
 * Fills a random matrix with given sparsity pattern.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 * @tparam IndexType  the type for row and column indices
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param row_idxs  the row indices of the matrix
 * @param col_idxs  the column indices of the matrix
 * @param value_dist  distribution of matrix values
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * @return the unique pointer of MatrixType
 */
template <typename MatrixType = matrix::Dense<>,
          typename IndexType = typename MatrixType::index_type,
          typename ValueDistribution, typename Engine, typename... MatrixArgs>
std::unique_ptr<MatrixType> fill_random_matrix(
    size_type num_rows, size_type num_cols,
    const gko::array<IndexType>& row_idxs,
    const gko::array<IndexType>& col_idxs, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = IndexType;

    GKO_ASSERT(row_idxs.get_size() == col_idxs.get_size());
    GKO_ASSERT(row_idxs.get_size() <= (num_rows * num_cols));
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(fill_random_matrix_data<value_type, index_type>(
        num_rows, num_cols, row_idxs, col_idxs,
        std::forward<ValueDistribution>(value_dist),
        std::forward<Engine>(engine)));
    return result;
}


/**
 * Generates a random matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param nonzero_dist  distribution of nonzeros per row
 * @param value_dist  distribution of matrix values
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
 * Generates a random dense matrix.
 *
 * @tparam ValueType  value type of the generated matrix
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param num_rows  number of rows
 * @param num_cols  number of columns
 * @param value_dist  distribution of matrix values
 * @param engine  a random engine
 * @param exec  executor where the matrix should be allocated
 * @param args  additional arguments for the matrix constructor
 *
 * @return the unique pointer of gko::matrix::Dense<ValueType>
 */
template <typename ValueType, typename ValueDistribution, typename Engine,
          typename... MatrixArgs>
std::unique_ptr<gko::matrix::Dense<ValueType>> generate_random_dense_matrix(
    size_type num_rows, size_type num_cols, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    auto result = gko::matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{num_rows, num_cols},
        std::forward<MatrixArgs>(args)...);
    result->read(
        matrix_data<ValueType, int>{gko::dim<2>{num_rows, num_cols},
                                    std::forward<ValueDistribution>(value_dist),
                                    std::forward<Engine>(engine)});
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
 * @param size  number of rows and columns
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
    size_type size, bool ones_on_diagonal, bool lower_triangular,
    NonzeroDistribution&& nonzero_dist, ValueDistribution&& value_dist,
    Engine&& engine)
{
    using std::begin;
    using std::end;

    matrix_data<ValueType, IndexType> data{gko::dim<2>{size, size}, {}};

    std::vector<bool> present_cols(size);

    for (IndexType row = 0; row < size; ++row) {
        // randomly generate number of nonzeros in this row
        const auto min_col = lower_triangular ? 0 : row;
        const auto max_col =
            lower_triangular ? row : static_cast<IndexType>(size) - 1;
        const auto max_row_nnz = max_col - min_col + 1;
        const auto nnz_in_row = std::max(
            size_type(0), std::min(static_cast<size_type>(nonzero_dist(engine)),
                                   static_cast<size_type>(max_row_nnz)));
        std::uniform_int_distribution<IndexType> col_dist{min_col, max_col};
        if (nnz_in_row > max_row_nnz / 2) {
            present_cols.assign(size, true);
            // remove max_row_nnz - nnz_in_row entries from present_cols
            size_type count = max_row_nnz;
            while (count > nnz_in_row) {
                const auto new_col = col_dist(engine);
                if (present_cols[new_col]) {
                    present_cols[new_col] = false;
                    count--;
                }
            }
            for (auto col = min_col; col <= max_col; col++) {
                if (present_cols[col] || col == row) {
                    data.nonzeros.emplace_back(
                        row, col,
                        row == col && ones_on_diagonal
                            ? one<ValueType>()
                            : detail::get_rand_value<ValueType>(value_dist,
                                                                engine));
                }
            }
        } else {
            // add nnz_in_row entries to present_cols
            present_cols.assign(size, false);
            size_type count = 0;
            while (count < nnz_in_row) {
                const auto col = col_dist(engine);
                if (!present_cols[col]) {
                    present_cols[col] = true;
                    count++;
                    data.nonzeros.emplace_back(
                        row, col,
                        row == col && ones_on_diagonal
                            ? one<ValueType>()
                            : detail::get_rand_value<ValueType>(value_dist,
                                                                engine));
                }
            }
            if (!present_cols[row]) {
                data.nonzeros.emplace_back(
                    row, row,
                    ones_on_diagonal ? one<ValueType>()
                                     : detail::get_rand_value<ValueType>(
                                           value_dist, engine));
            }
        }
    }

    data.sort_row_major();
    return data;
}


/**
 * Generates a random triangular matrix.
 *
 * @tparam MatrixType  type of matrix to generate (must implement
 *                     the interface `ReadableFromMatrixData<>` and provide
 *                     matching `value_type` and `index_type` type aliases)
 * @tparam NonzeroDistribution  type of nonzero distribution
 * @tparam ValueDistribution  type of value distribution
 * @tparam Engine  type of random engine
 *
 * @param size  number of rows and columns
 * @param ones_on_diagonal  `true` generates only ones on the diagonal,
 *                          `false` generates random values on the diagonal
 * @param lower_triangular  `true` generates a lower triangular matrix,
 *                          `false` an upper triangular matrix
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
std::unique_ptr<MatrixType> generate_random_triangular_matrix(
    size_type size, bool ones_on_diagonal, bool lower_triangular,
    NonzeroDistribution&& nonzero_dist, ValueDistribution&& value_dist,
    Engine&& engine, std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    using value_type = typename MatrixType::value_type;
    using index_type = typename MatrixType::index_type;
    auto result = MatrixType::create(exec, std::forward<MatrixArgs>(args)...);
    result->read(generate_random_triangular_matrix_data<value_type, index_type>(
        size, ones_on_diagonal, lower_triangular,
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
 * @param size  number of rows and columns
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
    size_type size, bool ones_on_diagonal, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    return generate_random_triangular_matrix<MatrixType>(
        size, ones_on_diagonal, true, nonzero_dist, value_dist, engine,
        std::move(exec), std::forward<MatrixArgs>(args)...);
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
 * @param size  number of rows and columns
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
    size_type size, bool ones_on_diagonal, NonzeroDistribution&& nonzero_dist,
    ValueDistribution&& value_dist, Engine&& engine,
    std::shared_ptr<const Executor> exec, MatrixArgs&&... args)
{
    return generate_random_triangular_matrix<MatrixType>(
        size, ones_on_diagonal, false, nonzero_dist, value_dist, engine,
        std::move(exec), std::forward<MatrixArgs>(args)...);
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


/**
 * Generates a tridiagonal Toeplitz matrix.
 *
 * @param size  the (square) size of the resulting matrix
 * @param coeffs  the coefficients of the tridiagonal matrix stored as [lower,
 *                diag, upper]
 * @param exec  the executor for the resulting matrix
 *
 * @return  a tridiagonal Toeplitz matrix.
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_tridiag_matrix_data(
    gko::size_type size, std::array<ValueType, 3> coeffs,
    std::shared_ptr<const gko::Executor> exec)
{
    auto lower = coeffs[0];
    auto diag = coeffs[1];
    auto upper = coeffs[2];

    gko::matrix_data<ValueType, IndexType> md{gko::dim<2>{size, size}};
    for (size_type i = 0; i < size; ++i) {
        if (i > 0) {
            md.nonzeros.emplace_back(i, i - 1, lower);
        }
        md.nonzeros.emplace_back(i, i, diag);
        if (i < size - 1) {
            md.nonzeros.emplace_back(i, i + 1, upper);
        }
    }
    return md;
}


/**
 * @copydoc generate_tridiag_matrix_data
 */
template <typename MatrixType>
std::unique_ptr<MatrixType> generate_tridiag_matrix(
    gko::size_type size, std::array<typename MatrixType::value_type, 3> coeffs,
    std::shared_ptr<const gko::Executor> exec)
{
    auto mtx = MatrixType::create(exec);
    mtx->read(generate_tridiag_matrix_data<typename MatrixType::value_type,
                                           typename MatrixType::index_type>(
        size, coeffs, exec));
    return mtx;
}


/**
 * This computes an inverse of an tridiagonal Toeplitz matrix.
 *
 * The computation is based on the formula is from
 * https://en.wikipedia.org/wiki/Tridiagonal_matrix#Inversion
 *
 * @param size  the (square) size of the resulting matrix
 * @param coeffs  the coefficients of the tridiagonal matrix stored as [lower,
 *                diag, upper]
 * @param exec  the executor for the resulting matrix
 *
 * @return  a matrix (possible dense) that is the inverse of the matrix
 *           generated from generate_tridiag_matrix_data with the same inputs
 */
template <typename ValueType, typename IndexType>
gko::matrix_data<ValueType, IndexType> generate_tridiag_inverse_matrix_data(
    gko::size_type size, std::array<ValueType, 3> coeffs)
{
    if (size == 0) {
        return {};
    }

    auto lower = coeffs[0];
    auto diag = coeffs[1];
    auto upper = coeffs[2];

    std::vector<ValueType> alpha(size + 1);
    auto beta = std::make_reverse_iterator(alpha.end());

    alpha[0] = 1;
    alpha[1] = diag;
    for (std::size_t i = 2; i < alpha.size(); ++i) {
        alpha[i] = diag * alpha[i - 1] - upper * lower * alpha[i - 2];
    }

    gko::matrix_data<ValueType, IndexType> md{gko::dim<2>{size, size}};
    for (size_type i = 0; i < size; ++i) {
        for (size_type j = 0; j < size; ++j) {
            if (i == j) {
                md.nonzeros.emplace_back(i, j,
                                         alpha[i] * beta[j + 1] / alpha.back());
            } else {
                auto sign = static_cast<ValueType>((i + j) % 2 ? -1 : 1);
                auto off_diag = i < j ? upper : lower;
                auto min_idx = std::min(i, j);
                auto max_idx = std::max(i, j);
                auto val = sign *
                           static_cast<ValueType>(
                               std::pow(off_diag, max_idx - min_idx)) *
                           alpha[min_idx] * beta[max_idx + 1] / alpha.back();
                md.nonzeros.emplace_back(i, j, val);
            }
        }
    }
    return md;
}


/**
 * @copydoc generate_tridiag_inverse_matrix_data
 */
template <typename MatrixType>
std::unique_ptr<MatrixType> generate_tridiag_inverse_matrix(
    gko::size_type size, std::array<typename MatrixType::value_type, 3> coeffs,
    std::shared_ptr<const gko::Executor> exec)
{
    auto mtx = MatrixType::create(exec);
    mtx->read(
        generate_tridiag_inverse_matrix_data<typename MatrixType::value_type,
                                             typename MatrixType::index_type>(
            size, coeffs));
    return mtx;
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_MATRIX_GENERATOR_HPP_
