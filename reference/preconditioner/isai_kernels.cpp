/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/preconditioner/isai_kernels.hpp"


#include <algorithm>
#include <memory>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Isai preconditioner namespace.
 *
 * @ingroup isai
 */
namespace isai {


template <typename ValueType, typename IndexType>
void generate_sparsity_l(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *l_csc,
                         matrix::Csr<ValueType, IndexType> *csc_sparsity)
{
    auto num_elems = l_csc->get_num_stored_elements();
    Array<IndexType> rows_array{exec, num_elems};
    Array<ValueType> vals_array{exec, num_elems};  // no need to initialize it

    const auto l_rows = l_csc->get_const_col_idxs();
    auto new_rows = rows_array.get_data();
    std::copy(l_rows, l_rows + num_elems, new_rows);

    const auto l_cols = l_csc->get_const_row_ptrs();
    auto new_cols = csc_sparsity->get_row_ptrs();  // was already allocated
    const auto num_cols = l_csc->get_size()[0];
    std::copy(l_cols, l_cols + num_cols, new_cols);

    matrix::CsrBuilder<ValueType, IndexType> builder{csc_sparsity};
    builder.get_col_idx_array() = std::move(rows_array);
    builder.get_value_array() = std::move(vals_array);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_SPARSITY_L_KERNEL);


template <typename ValueType, typename IndexType>
void generate_l(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType> *l_csc,
                matrix::Csr<ValueType, IndexType> *inverse_l)
{
    // Note: both matrices are in CSC format and not in CSR
    //       (basically a stored transposed CSR format)
    auto size = l_csc->get_size();
    const auto l_col_ptrs = l_csc->get_const_row_ptrs();
    const auto l_rows = l_csc->get_const_col_idxs();
    const auto l_vals = l_csc->get_const_values();
    auto inv_col_ptrs = inverse_l->get_row_ptrs();
    auto inv_rows = inverse_l->get_col_idxs();
    auto inv_vals = inverse_l->get_values();

    /*
    // TODO: Make this its own kernel!
    // Copy sparsity pattern of original L into the inverse of L
    inv_col_ptrs[0] = l_col_ptrs[0];
    for (IndexType col = 0; col < size[0]; ++col) {
        inv_col_ptrs[col] = l_col_ptrs[col];
        inv_rows[col] = l_rows[col];
    }
    */

    std::vector<ValueType> rhs;  // RHS for local trisystem
    // memory for dense trisystem in column major:
    std::vector<ValueType> trisystem;

    for (IndexType col = 0; col < size[0]; ++col) {
        const auto inv_col_begin = inv_col_ptrs[col];
        const auto inv_col_end = inv_col_ptrs[col + 1];
        const auto inv_col_elems = inv_col_end - inv_col_begin;

        trisystem.clear();
        trisystem.reserve(inv_col_elems * inv_col_elems);
        for (IndexType i = 0; i < inv_col_elems * inv_col_elems; ++i) {
            trisystem.push_back(zero<ValueType>());
        }

        // generate the triangular system:
        // Go over inv of the current row to get the sparsity pattern S to look
        // for. Search for the same columns as S in the rows S^T and store them
        // in the triangular System dense matrix (which is stored column major
        // for coalescend writes)
        for (IndexType i = 0; i < inv_col_elems; ++i) {
            const auto row = inv_rows[inv_col_begin + i];
            const auto l_col_end = l_col_ptrs[row + 1];
            auto l_col_ptr = l_col_ptrs[row];
            auto inv_col_ptr = inv_col_begin;
            auto idx = i * inv_col_elems;  // idx for triagsystem to write to
            while (l_col_ptr < l_col_end &&
                   inv_col_ptr <
                       inv_col_end) {  // stop once this column is done
                const auto sparsity_col = inv_rows[inv_col_ptr];
                const auto l_col = l_rows[l_col_ptr];
                if (sparsity_col == l_col) {  // match found
                    trisystem[idx] = l_vals[l_col_ptr];
                    ++l_col_ptr;
                    ++inv_col_ptr;
                    ++idx;
                } else if (l_col < sparsity_col) {
                    // Check next element in L
                    ++l_col_ptr;
                } else {
                    // element does not exist, i.e. inv_col_ptr <
                    // l_cols[l_col_ptr]
                    ++inv_col_ptr;  // check next elment in the sparsity pattern
                    ++idx;          // leave this element equal zero
                }
            }
        }

        // second: solve the triangular systems
        // Triangular solve. The dense matrix is in colmajor.
        rhs.clear();
        rhs.reserve(inv_col_elems);
        for (IndexType d_row = 0; d_row < inv_col_elems; ++d_row) {
            // RHS is identity: 1 first value, 0 the rest
            rhs.push_back(d_row == 0 ? one<ValueType>() : zero<ValueType>());
        }

        for (IndexType d_col = 0; d_col < inv_col_elems;
             ++d_col) {  // go over dense cols
            const auto diag = trisystem[d_col * inv_col_elems + d_col];
            const auto top = rhs[d_col] / diag;
            rhs[d_col] = top;
            for (IndexType d_row = d_col + 1; d_row < inv_col_elems;
                 ++d_row) {  // go over all dense rows
                rhs[d_row] -= top * trisystem[d_col * inv_col_elems + d_row];
            }
        }

        // Drop B to dev memory - in ISAI preconditioner M
        for (IndexType i = 0; i < inv_col_elems; ++i) {
            inv_vals[inv_col_begin + i] = rhs[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_L_KERNEL);


template <typename ValueType, typename IndexType>
void generate_sparsity_u(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *u_csc,
                         matrix::Csr<ValueType, IndexType> *csc_sparsity)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_SPARSITY_U_KERNEL);


template <typename ValueType, typename IndexType>
void generate_u(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType> *u_csc,
                matrix::Csr<ValueType, IndexType> *inverse_u)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_U_KERNEL);


}  // namespace isai
}  // namespace reference
}  // namespace kernels
}  // namespace gko
