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

#include "core/matrix/dense_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "core/components/prefix_sum.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *a,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *c)
{
#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type col = 0; col < c->get_size()[1]; ++col) {
            c->at(row, col) = zero<ValueType>();
        }
    }

#pragma omp parallel for
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
void apply(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *a, const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *c)
{
    if (beta->at(0, 0) != zero<ValueType>()) {
#pragma omp parallel for
        for (size_type row = 0; row < c->get_size()[0]; ++row) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) *= beta->at(0, 0);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type row = 0; row < c->get_size()[0]; ++row) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) *= zero<ValueType>();
            }
        }
    }

#pragma omp parallel for
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
void scale(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *x)
{
    if (alpha->get_size()[1] == 1) {
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                x->at(i, j) *= alpha->at(0, 0);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                x->at(i, j) *= alpha->at(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const OmpExecutor> exec,
                const matrix::Dense<ValueType> *alpha,
                const matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *y)
{
    if (alpha->get_size()[1] == 1) {
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                y->at(i, j) += alpha->at(0, 0) * x->at(i, j);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < x->get_size()[1]; ++j) {
                y->at(i, j) += alpha->at(0, j) * x->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const OmpExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Diagonal<ValueType> *x,
                     matrix::Dense<ValueType> *y)
{
    const auto diag_values = x->get_const_values();
#pragma omp parallel for
    for (size_type i = 0; i < x->get_size()[0]; i++) {
        y->at(i, i) += alpha->at(0, 0) * diag_values[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const OmpExecutor> exec,
                 const matrix::Dense<ValueType> *x,
                 const matrix::Dense<ValueType> *y,
                 matrix::Dense<ValueType> *result)
{
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        result->at(0, j) = zero<ValueType>();
    }
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            result->at(0, j) += conj(x->at(i, j)) * y->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType> *x,
                   matrix::Dense<remove_complex<ValueType>> *result)
{
    using norm_type = remove_complex<ValueType>;
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        result->at(0, j) = zero<norm_type>();
    }
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            result->at(0, j) += squared_norm(x->at(i, j));
        }
    }
#pragma omp parallel for
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        result->at(0, j) = sqrt(result->at(0, j));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    matrix::Coo<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();
    Array<IndexType> row_ptrs_array(exec, num_rows);
    auto row_ptrs = row_ptrs_array.get_data();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType row_count{};
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            row_count += val != zero<ValueType>();
        }
        row_ptrs[row] = row_count;
    }

    components::prefix_sum(exec, row_ptrs, num_rows);

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto idxs = row_ptrs[row];
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
void convert_to_csr(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType row_nnz{};
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            row_nnz += val != zero<ValueType>();
        }
        row_ptrs[row] = row_nnz;
    }

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto cur_ptr = row_ptrs[row];
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = col;
                values[cur_ptr] = val;
                ++cur_ptr;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *source,
                    matrix::Ell<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();
#pragma omp parallel for
    for (size_type i = 0; i < max_nnz_per_row; i++) {
        for (size_type j = 0; j < result->get_stride(); j++) {
            result->val_at(j, i) = zero<ValueType>();
            result->col_at(j, i) = 0;
        }
    }
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        size_type col_idx = 0;
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
void convert_to_hybrid(std::shared_ptr<const OmpExecutor> exec,
                       const matrix::Dense<ValueType> *source,
                       matrix::Hybrid<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();
    Array<IndexType> coo_row_ptrs_array(exec, num_rows);
    auto coo_row_ptrs = coo_row_ptrs_array.get_data();

    auto ell_nnz_row = result->get_ell_num_stored_elements_per_row();
    auto ell_stride = result->get_ell_stride();
#pragma omp parallel for collapse(2)
    for (size_type i = 0; i < ell_nnz_row; i++) {
        for (size_type j = 0; j < ell_stride; j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = 0;
        }
    }
#pragma omp parallel for
    for (size_type i = 0; i < result->get_coo_num_stored_elements(); i++) {
        coo_val[i] = zero<ValueType>();
        coo_col[i] = 0;
        coo_row[i] = 0;
    }
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        size_type total_row_nnz{};
        for (size_type col = 0; col < num_cols; col++) {
            auto val = source->at(row, col);
            total_row_nnz += val != zero<ValueType>();
        }
        coo_row_ptrs[row] = std::max(ell_lim, total_row_nnz) - ell_lim;
    }

    components::prefix_sum(exec, coo_row_ptrs, num_rows);

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        size_type ell_count = 0;
        size_type col = 0;
        for (; col < num_cols && ell_count < ell_lim; col++) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                result->ell_val_at(row, ell_count) = val;
                result->ell_col_at(row, ell_count) = col;
                ell_count++;
            }
        }
        auto coo_idx = coo_row_ptrs[row];
        for (; col < num_cols; col++) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                coo_val[coo_idx] = val;
                coo_col[coo_idx] = col;
                coo_row[coo_idx] = row;
                coo_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Dense<ValueType> *source,
                      matrix::Sellp<ValueType, IndexType> *result)
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
        size_type current_slice_length = 0;
#pragma omp parallel for reduction(max : current_slice_length)
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row < num_rows) {
                size_type max_col = 0;
                for (size_type col = 0; col < num_cols; col++) {
                    if (source->at(global_row, col) != zero<ValueType>()) {
                        max_col += 1;
                    }
                }
                current_slice_length = std::max(current_slice_length, max_col);
            }
        }
        slice_lengths[slice] =
            stride_factor * ceildiv(current_slice_length, stride_factor);
#pragma omp parallel for
        for (size_type row = 0; row < slice_size; row++) {
            const size_type global_row = slice * slice_size + row;
            if (global_row < num_rows) {
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
                     (slice_sets[slice] + slice_lengths[slice]) * slice_size +
                         row;
                     i += slice_size) {
                    col_idxs[i] = 0;
                    vals[i] = 0;
                }
            }
        }
    }

    if (slice_num > 0) {
        slice_sets[slice_num] =
            slice_sets[slice_num - 1] + slice_lengths[slice_num - 1];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const OmpExecutor> exec,
                             const matrix::Dense<ValueType> *source,
                             matrix::SparsityCsr<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto value = result->get_value();
    value[0] = one<ValueType>();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType row_nnz{};
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            row_nnz += val != zero<ValueType>();
        }
        row_ptrs[row] = row_nnz;
    }

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto cur_ptr = row_ptrs[row];
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = col;
                ++cur_ptr;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *source, size_type *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto num_nonzeros = 0;

#pragma omp parallel for reduction(+ : num_nonzeros)
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
    }

    *result = num_nonzeros;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const OmpExecutor> exec,
                               const matrix::Dense<ValueType> *source,
                               size_type *result)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    size_type max_nonzeros_per_row = 0;
#pragma omp parallel for reduction(max : max_nonzeros_per_row)
    for (size_type row = 0; row < num_rows; ++row) {
        size_type num_nonzeros = 0;
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += (source->at(row, col) != zero<ValueType>());
        }
        max_nonzeros_per_row = std::max(num_nonzeros, max_nonzeros_per_row);
    }
    *result = max_nonzeros_per_row;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const OmpExecutor> exec,
                                const matrix::Dense<ValueType> *source,
                                Array<size_type> *result)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto row_nnz_val = result->get_data();
#pragma omp parallel for
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
void calculate_total_cols(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::Dense<ValueType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size)
{
    auto num_rows = source->get_size()[0];
    auto num_cols = source->get_size()[1];
    auto slice_num = ceildiv(num_rows, slice_size);
    size_type total_cols = 0;
#pragma omp parallel for reduction(+ : total_cols)
    for (size_type slice = 0; slice < slice_num; slice++) {
        size_type slice_temp = 0;
        for (size_type row = 0;
             row < slice_size && row + slice * slice_size < num_rows; row++) {
            size_type temp = 0;
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
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::Dense<ValueType> *orig,
               matrix::Dense<ValueType> *trans)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            trans->at(j, i) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType> *orig,
                    matrix::Dense<ValueType> *trans)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            trans->at(j, i) = conj(orig->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void row_gather(std::shared_ptr<const OmpExecutor> exec,
                const Array<IndexType> *row_indices,
                const matrix::Dense<ValueType> *orig,
                matrix::Dense<ValueType> *row_gathered)
{
    auto rows = row_indices->get_const_data();
#pragma omp parallel for
    for (size_type i = 0; i < row_indices->get_num_elems(); ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            row_gathered->at(i, j) = orig->at(rows[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_ROW_GATHER_KERNEL);


template <typename ValueType, typename IndexType>
void column_permute(std::shared_ptr<const OmpExecutor> exec,
                    const Array<IndexType> *permutation_indices,
                    const matrix::Dense<ValueType> *orig,
                    matrix::Dense<ValueType> *column_permuted)
{
    auto perm = permutation_indices->get_const_data();
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            column_permuted->at(i, j) = orig->at(i, perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COLUMN_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_row_permute(std::shared_ptr<const OmpExecutor> exec,
                         const Array<IndexType> *permutation_indices,
                         const matrix::Dense<ValueType> *orig,
                         matrix::Dense<ValueType> *row_permuted)
{
    auto perm = permutation_indices->get_const_data();
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            row_permuted->at(perm[i], j) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inverse_column_permute(std::shared_ptr<const OmpExecutor> exec,
                            const Array<IndexType> *permutation_indices,
                            const matrix::Dense<ValueType> *orig,
                            matrix::Dense<ValueType> *column_permuted)
{
    auto perm = permutation_indices->get_const_data();
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            column_permuted->at(i, perm[j]) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_COLUMN_PERMUTE_KERNEL);


template <typename ValueType>
void extract_diagonal(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Dense<ValueType> *orig,
                      matrix::Diagonal<ValueType> *diag)
{
    auto diag_values = diag->get_values();
#pragma omp parallel for
    for (size_type i = 0; i < diag->get_size()[0]; ++i) {
        diag_values[i] = orig->at(i, i);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType>
void inplace_absolute_dense(std::shared_ptr<const OmpExecutor> exec,
                            matrix::Dense<ValueType> *source)
{
    auto dim = source->get_size();

#pragma omp parallel for
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            source->at(row, col) = abs(source->at(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void outplace_absolute_dense(std::shared_ptr<const OmpExecutor> exec,
                             const matrix::Dense<ValueType> *source,
                             matrix::Dense<remove_complex<ValueType>> *result)
{
    auto dim = source->get_size();

#pragma omp parallel for
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result->at(row, col) = abs(source->at(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void make_complex(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Dense<ValueType> *source,
                  matrix::Dense<to_complex<ValueType>> *result)
{
    auto dim = source->get_size();

#pragma omp parallel for
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result->at(row, col) = to_complex<ValueType>{source->at(row, col)};
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MAKE_COMPLEX_KERNEL);


template <typename ValueType>
void get_real(std::shared_ptr<const OmpExecutor> exec,
              const matrix::Dense<ValueType> *source,
              matrix::Dense<remove_complex<ValueType>> *result)
{
    auto dim = source->get_size();

#pragma omp parallel for
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result->at(row, col) = real(source->at(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_REAL_KERNEL);


template <typename ValueType>
void get_imag(std::shared_ptr<const OmpExecutor> exec,
              const matrix::Dense<ValueType> *source,
              matrix::Dense<remove_complex<ValueType>> *result)
{
    auto dim = source->get_size();

#pragma omp parallel for
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result->at(row, col) = imag(source->at(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_IMAG_KERNEL);


}  // namespace dense
}  // namespace omp
}  // namespace kernels
}  // namespace gko
