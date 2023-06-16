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

#ifndef GKO_CORE_UTILS_MATRIX_UTILS_HPP_
#define GKO_CORE_UTILS_MATRIX_UTILS_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/temporary_clone.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace utils {


/**
 * Removes all strictly upper triangular entries from the given matrix data.
 *
 * @param data  the matrix data
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_lower_triangular(matrix_data<ValueType, IndexType>& data)
{
    data.nonzeros.erase(
        std::remove_if(data.nonzeros.begin(), data.nonzeros.end(),
                       [](auto entry) { return entry.row < entry.column; }),
        data.nonzeros.end());
}


/**
 * Removes all strictly lower triangular entries from the given matrix data.
 *
 * @param data  the matrix data
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_upper_triangular(matrix_data<ValueType, IndexType>& data)
{
    data.nonzeros.erase(
        std::remove_if(data.nonzeros.begin(), data.nonzeros.end(),
                       [](auto entry) { return entry.row > entry.column; }),
        data.nonzeros.end());
}


/**
 * Removes all diagonal entries from the given matrix data.
 *
 * @param data  the matrix data
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_remove_diagonal(matrix_data<ValueType, IndexType>& data)
{
    data.nonzeros.erase(
        std::remove_if(data.nonzeros.begin(), data.nonzeros.end(),
                       [](auto entry) { return entry.row == entry.column; }),
        data.nonzeros.end());
}


/**
 * Sets all diagonal entries for the given matrix data to one.
 *
 * @param data  the matrix data
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_unit_diagonal(matrix_data<ValueType, IndexType>& data)
{
    make_remove_diagonal(data);
    auto num_diags = std::min(data.size[0], data.size[1]);
    for (gko::int64 i = 0; i < num_diags; i++) {
        data.nonzeros.emplace_back(i, i, one<ValueType>());
    }
    data.ensure_row_major_order();
}


/**
 * Replace A by (A + op(A^T)) / 2 in the matrix data
 *
 * @param data  the matrix data
 * @param op  the operator to apply to the transposed value
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType, typename Op>
void make_symmetric_generic(matrix_data<ValueType, IndexType>& data, Op op)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(data.size);
    // compute (A + op(A^T)) / 2
    const auto nnz = data.nonzeros.size();
    for (std::size_t i = 0; i < nnz; i++) {
        data.nonzeros[i].value /= 2.0;
        auto entry = data.nonzeros[i];
        data.nonzeros.emplace_back(entry.column, entry.row, op(entry.value));
    }
    // combine duplicates
    data.sum_duplicates();
}


/**
 * Replace A by (A + A^T) / 2 in the matrix data
 *
 * @param data  the matrix data
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_symmetric(matrix_data<ValueType, IndexType>& data)
{
    make_symmetric_generic(data, [](auto val) { return val; });
}


/**
 * Replace A by (A + conj(A^T)) / 2 in the matrix data
 *
 * @param data  the matrix data
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_hermitian(matrix_data<ValueType, IndexType>& data)
{
    make_symmetric_generic(data, [](auto val) { return gko::conj(val); });
}


/**
 * Scales the matrix data such that its diagonal is at least ratio times as
 * large as the sum of offdiagonal magnitudes. Inserts diagonals if missing.
 *
 * @param data  the matrix data
 * @param ratio  how much bigger should the diagonal entries be compared to the
 *               offdiagonal entry magnitude sum?
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_diag_dominant(matrix_data<ValueType, IndexType>& data,
                        remove_complex<ValueType> ratio = 1.0)
{
    GKO_ASSERT_EQ(ratio >= 1.0, true);
    GKO_ASSERT_IS_SQUARE_MATRIX(data.size);
    std::vector<remove_complex<ValueType>> norms(data.size[0]);
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
    for (i = 0; i < data.size[0]; i++) {
        if (norms[i] == zero<ValueType>()) {
            // make sure empty rows don't make the matrix singular
            norms[i] = one<remove_complex<ValueType>>();
        }
        if (diag_positions[i] < 0) {
            data.nonzeros.emplace_back(i, i, norms[i] * ratio);
        } else {
            auto& diag_value = data.nonzeros[diag_positions[i]].value;
            const auto diag_magnitude = gko::abs(diag_value);
            const auto offdiag_magnitude = norms[i];
            if (diag_magnitude < offdiag_magnitude * ratio) {
                const auto scaled_value =
                    diag_value * (offdiag_magnitude * ratio / diag_magnitude);
                if (gko::is_finite(scaled_value)) {
                    diag_value = scaled_value;
                } else {
                    diag_value = offdiag_magnitude * ratio;
                }
            }
        }
    }
    data.ensure_row_major_order();
}


/**
 * Makes the matrix symmetric and diagonally dominant by the given ratio.
 *
 * @param data  the matrix data
 * @param ratio  how much bigger should the diagonal entries be compared to the
 *               offdiagonal entry magnitude sum?
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_spd(matrix_data<ValueType, IndexType>& data,
              remove_complex<ValueType> ratio = 1.001)
{
    GKO_ASSERT_EQ(ratio > 1.0, true);
    make_symmetric(data);
    make_diag_dominant(data, ratio);
}


/**
 * Makes the matrix hermitian and diagonally dominant by the given ratio.
 *
 * @param data  the matrix data
 * @param ratio  how much bigger should the diagonal entries be compared to the
 *               offdiagonal entry magnitude sum?
 *
 * @tparam ValueType  the value type
 * @tparam IndexType  the index type
 */
template <typename ValueType, typename IndexType>
void make_hpd(matrix_data<ValueType, IndexType>& data,
              remove_complex<ValueType> ratio = 1.001)
{
    GKO_ASSERT_EQ(ratio > 1.0, true);
    make_hermitian(data);
    make_diag_dominant(data, ratio);
}


/**
 * Changes the diagonal entry in the requested row, shrinking the
 * matrix by 1 nonzero entry.
 *
 * @param mtx  The matrix to remove a diagonal entry from.
 * @param row_to_process  The row from which to remove the diagonal entry.
 */
template <typename MtxType>
void remove_diagonal_entry_from_row(
    MtxType* mtx, const typename MtxType::index_type row_to_process)
{
    using value_type = typename MtxType::value_type;
    using index_type = typename MtxType::index_type;
    matrix_data<value_type, index_type> mdata;
    mtx->write(mdata);
    auto it = std::remove_if(mdata.nonzeros.begin(), mdata.nonzeros.end(),
                             [&](auto entry) {
                                 return entry.row == row_to_process &&
                                        entry.column == row_to_process;
                             });
    mdata.nonzeros.erase(it, mdata.nonzeros.end());
    mtx->read(mdata);
}


/**
 * Ensures each row has a diagonal entry.
 */
template <typename MtxType>
void ensure_all_diagonal_entries(MtxType* mtx)
{
    using value_type = typename MtxType::value_type;
    using index_type = typename MtxType::index_type;
    matrix_data<value_type, index_type> mdata;
    mtx->write(mdata);
    const auto nrows = static_cast<index_type>(mtx->get_size()[0]);
    mdata.nonzeros.reserve(mtx->get_num_stored_elements() + nrows);
    for (index_type i = 0; i < nrows; i++) {
        mdata.nonzeros.push_back({i, i, zero<value_type>()});
    }
    mdata.sum_duplicates();
    mtx->read(mdata);
}


}  // namespace utils
}  // namespace gko

#endif  // GKO_CORE_UTILS_MATRIX_UTILS_HPP_
