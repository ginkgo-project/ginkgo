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
 *      functions which do not have to be optimized for performance.
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
    using nonzero_type = std::tuple<IndexType, IndexType, ValueType>;


    matrix_data() = default;

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
            nonzeros_ = {})
        : num_rows(num_rows_), num_cols(num_cols_), nonzeros()
    {
        nonzeros.reserve(nonzeros_.size());
        for (const auto &elem : nonzeros_) {
            nonzeros.emplace_back(elem.row, elem.col, elem.val);
        }
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
        std::sort(begin(nonzeros), end(nonzeros),
                  [](nonzero_type x, nonzero_type y) {
                      return std::tie(std::get<0>(x), std::get<1>(x)) <
                             std::tie(std::get<0>(y), std::get<1>(y));
                  });
    }
};


}  // namespace gko


#endif  // GKO_CORE_BASE_MATRIX_DATA_HPP_
