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

#ifndef GKO_CORE_BASE_MATRIX_DATA_HPP_
#define GKO_CORE_BASE_MATRIX_DATA_HPP_


#include "core/base/math.hpp"
#include "core/base/types.hpp"


#include <algorithm>
#include <tuple>
#include <vector>


namespace gko {


namespace detail {


// internal structure used to get around explicit constructors in std::tuple
template <typename ValueType, typename IndexType>
struct input_triple {
    IndexType row;
    IndexType col;
    ValueType val;
};


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
 * This structure is used as an intermediate data type to store a sparse matrix.
 *
 * The matrix is stored as a sequence of nonzero elements, where each element is
 * a triple of the form (row_index, column_index, value).
 *
 * @note All Ginkgo functions returning such a structure will return the
 *       nonzeros sorted in row-major order.
 * @note All Ginkgo functions that take this structure as input expect that the
 *       nonzeros are sorted in row-major order.
 * @note This structure is not optimized for usual access patterns and it can
 *       only exist on the CPU. Thus, it should only be used for utility
 *       functions which do not have to be optimized for performance.
 *
 * @tparam ValueType  type of matrix values stored in the structure
 * @tparam IndexType  type of matrix indexes stored in the structure
 */
template <typename ValueType = default_precision, typename IndexType = int32>
struct matrix_data {
    using value_type = ValueType;
    using index_type = IndexType;

    /**
     * Type used to store nonzeros.
     */
    struct nonzero_type {
        nonzero_type() = default;

        nonzero_type(index_type r, index_type c, value_type v)
            : row(r), column(c), value(v)
        {}

#define GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(_op)                \
    bool operator _op(const nonzero_type &other) const          \
    {                                                           \
        return std::tie(this->row, this->column, this->value)   \
            _op std::tie(other.row, other.column, other.value); \
    }

        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(==);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(!=)
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(<);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(>);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(<=);
        GKO_DEFINE_DEFAULT_COMPARE_OPERATOR(>=);

#undef GKO_DEFINE_DEFAULT_COMPARE_OPERATOR

        index_type row;
        index_type column;
        value_type value;
    };


    /**
     * Initializes a 0-by-0 matrix.
     */
    matrix_data() : num_rows{}, num_cols{} {}

    /**
     * Initializes a matrix filled with the specified value.
     *
     * @param num_rows_  number of rows of the matrix
     * @param num_cols_  number of columns of the matrix
     * @param value  value used to fill the elements of the matrix
     */
    matrix_data(size_type num_rows_, size_type num_cols_,
                ValueType value = zero<ValueType>())
        : num_rows(num_rows_), num_cols(num_cols_)
    {
        if (value == zero<ValueType>()) {
            return;
        }
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                nonzeros.emplace_back(row, col, value);
            }
        }
    }

    /**
     * Initializes a matrix with random values from the specified distribution.
     *
     * @tparam RandomDistribution  random distribution type
     * @tparam RandomEngine  random engine type
     *
     * @param num_rows_  number of rows of the matrix
     * @param num_cols_  number of columns of the matrix
     * @param dist  random distribution of the elements of the matrix
     * @param engine  random engine used to generate random values
     */
    template <typename RandomDistribution, typename RandomEngine>
    matrix_data(size_type num_rows_, size_type num_cols_,
                RandomDistribution &&dist, RandomEngine &&engine)
        : num_rows(num_rows_), num_cols(num_cols_)
    {
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                const auto value =
                    detail::get_rand_value<ValueType>(dist, engine);
                if (value != zero<ValueType>()) {
                    nonzeros.emplace_back(row, col, value);
                }
            }
        }
    }

    /**
     * List-initializes the structure from a matrix of values.
     *
     * @param values  a 2D braced-init-list of matrix values.
     */
    matrix_data(std::initializer_list<std::initializer_list<ValueType>> values)
        : num_rows(values.size()), num_cols{}
    {
        for (size_type row = 0; row < values.size(); ++row) {
            const auto row_data = begin(values)[row];
            num_cols = std::max(num_cols, row_data.size());
            for (size_type col = 0; col < row_data.size(); ++col) {
                const auto &val = begin(row_data)[col];
                if (val != zero<ValueType>()) {
                    nonzeros.emplace_back(row, col, val);
                }
            }
        }
    }

    /**
     * Initializes the structure from a list of nonzeros.
     *
     * @param num_rows_  number of rows of the matrix
     * @param num_cols_  number of columns of the matrix
     * @param nonzeros_  list of nonzero elements
     */
    matrix_data(
        size_type num_rows_, size_type num_cols_,
        std::initializer_list<detail::input_triple<ValueType, IndexType>>
            nonzeros_)
        : num_rows(num_rows_), num_cols(num_cols_), nonzeros()
    {
        nonzeros.reserve(nonzeros_.size());
        for (const auto &elem : nonzeros_) {
            nonzeros.emplace_back(elem.row, elem.col, elem.val);
        }
    }

    /**
     * Initializes a diagonal matrix.
     *
     * @param num_rows_  number of rows of the matrix
     * @param num_cols_  number of columns of the matrix
     * @param value  value used to fill the elements of the matrix
     */
    static matrix_data diag(size_type num_rows, size_type num_cols,
                            ValueType value)
    {
        matrix_data res(num_rows, num_cols);
        if (value != zero<ValueType>()) {
            const auto num_nnz = std::min(num_rows, num_cols);
            res.nonzeros.reserve(num_nnz);
            for (int i = 0; i < num_nnz; ++i) {
                res.nonzeros.emplace_back(i, i, value);
            }
        }
        return res;
    }

    /**
     * Initializes a diagonal matrix using a list of diagonal elements.
     *
     * @param num_rows_  number of rows of the matrix
     * @param num_cols_  number of columns of the matrix
     * @param nonzeros_  list of diagonal elements
     */
    static matrix_data diag(size_type num_rows, size_type num_cols,
                            std::initializer_list<ValueType> nonzeros_)
    {
        matrix_data res(num_rows, num_cols);
        res.nonzeros.reserve(nonzeros_.size());
        int pos = 0;
        for (auto value : nonzeros_) {
            res.nonzeros.emplace_back(pos, pos, value);
            ++pos;
        }
        return res;
    }

    /**
     * Initializes a block-diagonal matrix.
     *
     * @param num_block_rows  number of block-rows
     * @param num_block_cols  number of block-columns
     * @param diag_block  matrix used to fill diagonal blocks
     */
    static matrix_data diag(size_type num_block_rows, size_type num_block_cols,
                            const matrix_data &block)
    {
        matrix_data res(num_block_rows * block.num_rows,
                        num_block_cols * block.num_cols);
        const auto num_blocks = std::min(num_block_rows, num_block_cols);
        res.nonzeros.reserve(num_blocks * block.nonzeros.size());
        for (int b = 0; b < num_blocks; ++b) {
            for (const auto &elem : block.nonzeros) {
                res.nonzeros.emplace_back(b * block.num_rows + elem.row,
                                          b * block.num_cols + elem.column,
                                          elem.value);
            }
        }
        return res;
    }

    /**
     * Total number of rows of the matrix.
     */
    size_type num_rows;

    /**
     * Total number of columns of the matrix.
     */
    size_type num_cols;

    /**
     * A vector of tuples storing the non-zeros of the matrix.
     *
     * The first two elements of the tuple are the row index and the column
     * index of a matrix element, and its third element is the value at that
     * position.
     */
    std::vector<nonzero_type> nonzeros;

    /**
     * Sorts the nonzero vector so the values follow row-major order.
     */
    void ensure_row_major_order()
    {
        std::sort(
            begin(nonzeros), end(nonzeros), [](nonzero_type x, nonzero_type y) {
                return std::tie(x.row, x.column) < std::tie(y.row, y.column);
            });
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MATRIX_DATA_HPP_
