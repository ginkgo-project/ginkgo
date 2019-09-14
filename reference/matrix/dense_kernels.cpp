/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/matrix/dense_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include <algorithm>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Dense matrix format namespace.
 * @ref Dense
 * @ingroup dense
 */
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type col = 0; col < c->get_size()[1]; ++col) {
            c->at(row, col) = zero<ValueType>();
        }
    }

    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type inner = 0; inner < a->get_size()[1]; ++inner) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) += a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    if (beta->at(0, 0) != zero<ValueType>()) {
        for (size_type row = 0; row < c->get_size()[0]; ++row) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) *= beta->at(0, 0);
            }
        }
    } else {
        for (size_type row = 0; row < c->get_size()[0]; ++row) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) *= zero<ValueType>();
            }
        }
    }

    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type inner = 0; inner < a->get_size()[1]; ++inner) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) +=
                    alpha->at(0, 0) * a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (alpha->get_size()[1] == 1) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                x->at(i, j) *= alpha->at(0, 0);
            }
        }
    } else {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                x->at(i, j) *= alpha->at(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (alpha->get_size()[1] == 1) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                y->at(i, j) += alpha->at(0, 0) * x->at(i, j);
            }
        }
    } else {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                y->at(i, j) += alpha->at(0, j) * x->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const ReferenceExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        result->at(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            result->at(0, j) += conj(x->at(i, j)) * y->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType> *x,
                   matrix::Dense<ValueType> *result)
{
    compute_dot(exec, x, x, result);
    for (size_type i = 0; i < result->get_size()[0]; ++i) {
        for (size_type j = 0; j < result->get_size()[1]; ++j) {
            result->at(i, j) = sqrt(abs(result->at(i, j)));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Coo<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    auto idxs = 0;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                row_idxs[idxs] = row;
                col_idxs[idxs] = col;
                values[idxs] = val;
                ++idxs;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    size_type cur_ptr = 0;
    row_ptrs[0] = cur_ptr;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = col;
                values[cur_ptr] = val;
                ++cur_ptr;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Ell<ValueType, IndexType> *result,
                    const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();
    for (size_type i = 0; i < max_nnz_per_row; i++) {
        for (size_type j = 0; j < result->get_stride(); j++) {
            result->val_at(j, i) = zero<ValueType>();
            result->col_at(j, i) = 0;
        }
    }
    size_type col_idx = 0;
    for (size_type row = 0; row < num_rows; row++) {
        col_idx = 0;
        for (size_type col = 0; col < num_cols; col++) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                result->val_at(row, col_idx) = val;
                result->col_at(row, col_idx) = col;
                col_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const ReferenceExecutor> exec,
                       matrix::Hybrid<ValueType, IndexType> *result,
                       const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
    auto coo_lim = strategy->get_coo_nnz();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();
    for (size_type i = 0; i < result->get_ell_num_stored_elements_per_row();
         i++) {
        for (size_type j = 0; j < result->get_ell_stride(); j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = 0;
        }
    }
    for (size_type i = 0; i < result->get_coo_num_stored_elements(); i++) {
        coo_val[i] = zero<ValueType>();
        coo_col[i] = 0;
        coo_row[i] = 0;
    }

    size_type coo_idx = 0;
    for (size_type row = 0; row < num_rows; row++) {
        size_type col_idx = 0, col = 0;
        while (col < num_cols && col_idx < ell_lim) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                result->ell_val_at(row, col_idx) = val;
                result->ell_col_at(row, col_idx) = col;
                col_idx++;
            }
            col++;
        }
        while (col < num_cols) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                coo_val[coo_idx] = val;
                coo_col[coo_idx] = col;
                coo_row[coo_idx] = row;
                coo_idx++;
            }
            col++;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::Sellp<ValueType, IndexType> *result,
                      const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();
    auto slice_size = (result->get_slice_size() == 0)
                          ? matrix::default_slice_size
                          : result->get_slice_size();
    auto stride_factor = (result->get_stride_factor() == 0)
                             ? matrix::default_stride_factor
                             : result->get_stride_factor();
    int slice_num = ceildiv(num_rows, slice_size);
    slice_sets[0] = 0;
    for (size_type slice = 0; slice < slice_num; slice++) {
        if (slice > 0) {
            slice_sets[slice] =
                slice_sets[slice - 1] + slice_lengths[slice - 1];
        }
        slice_lengths[slice] = 0;
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            size_type max_col = 0;
            for (size_type col = 0; col < num_cols; col++) {
                if (source->at(global_row, col) != zero<ValueType>()) {
                    max_col += 1;
                }
            }
            slice_lengths[slice] = std::max(slice_lengths[slice], max_col);
        }
        slice_lengths[slice] =
            stride_factor * ceildiv(slice_lengths[slice], stride_factor);
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row >= num_rows) {
                break;
            }
            size_type sellp_ind = slice_sets[slice] * slice_size + row;
            for (size_type col = 0; col < num_cols; col++) {
                auto val = source->at(global_row, col);
                if (val != zero<ValueType>()) {
                    col_idxs[sellp_ind] = col;
                    vals[sellp_ind] = val;
                    sellp_ind += slice_size;
                }
            }
            for (size_type i = sellp_ind;
                 i <
                 (slice_sets[slice] + slice_lengths[slice]) * slice_size + row;
                 i += slice_size) {
                col_idxs[i] = 0;
                vals[i] = 0;
            }
        }
    }
    slice_sets[slice_num] =
        slice_sets[slice_num - 1] + slice_lengths[slice_num - 1];
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const ReferenceExecutor> exec,
                             matrix::SparsityCsr<ValueType, IndexType> *result,
                             const matrix::Dense<ValueType> *source)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto value = result->get_value();
    value[0] = one<ValueType>();
    size_type cur_ptr = 0;
    row_ptrs[0] = cur_ptr;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = col;
                ++cur_ptr;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto num_nonzeros = 0;

    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
    }

    *result = num_nonzeros;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::Dense<ValueType> *source,
                               size_type *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    size_type num_stored_elements_per_row = 0;
    size_type num_nonzeros = 0;
    for (size_type row = 0; row < num_rows; ++row) {
        num_nonzeros = 0;
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
        num_stored_elements_per_row =
            std::max(num_nonzeros, num_stored_elements_per_row);
    }

    *result = num_stored_elements_per_row;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                                const matrix::Dense<ValueType> *source,
                                Array<size_type> *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_nnz_val = result->get_data();
    for (size_type row = 0; row < num_rows; ++row) {
        size_type num_nonzeros = 0;
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
        row_nnz_val[row] = num_nonzeros;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::Dense<ValueType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto slice_num = ceildiv(num_rows, slice_size);
    auto total_cols = 0;
    auto temp = 0, slice_temp = 0;
    for (size_type slice = 0; slice < slice_num; slice++) {
        slice_temp = 0;
        for (size_type row = 0;
             row < slice_size && row + slice * slice_size < num_rows; row++) {
            temp = 0;
            for (size_type col = 0; col < num_cols; col++) {
                temp += (source->at(row + slice * slice_size, col) !=
                         zero<ValueType>());
            }
            slice_temp = (slice_temp < temp) ? temp : slice_temp;
        }
        slice_temp = ceildiv(slice_temp, stride_factor) * stride_factor;
        total_cols += slice_temp;
    }

    *result = total_cols;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::Dense<ValueType> *trans,
               const matrix::Dense<ValueType> *orig)
{
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            trans->at(j, i) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::Dense<ValueType> *trans,
                    const matrix::Dense<ValueType> *orig)
{
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            trans->at(j, i) = conj(orig->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                 const Array<IndexType> *permutation_indices,
                 matrix::Dense<ValueType> *row_permuted,
                 const matrix::Dense<ValueType> *orig)
{
    auto perm = permutation_indices->get_const_data();
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            row_permuted->at(i, j) = orig->at(perm[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const ReferenceExecutor> exec,
                    const Array<IndexType> *permutation_indices,
                    matrix::Dense<ValueType> *column_permuted,
                    const matrix::Dense<ValueType> *orig)
{
    auto perm = permutation_indices->get_const_data();
    for (size_type j = 0; j < orig->get_size()[1]; ++j) {
        for (size_type i = 0; i < orig->get_size()[0]; ++i) {
            column_permuted->at(i, j) = orig->at(i, perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COLUMN_PERMUTE_KERNEL);


}  // namespace dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
