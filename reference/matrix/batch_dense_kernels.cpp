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

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_diagonal.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_dense
 */
namespace batch_dense {


#include "reference/matrix/batch_dense_kernels.hpp.inc"


template <typename ValueType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::BatchDense<ValueType>* const a,
                  const matrix::BatchDense<ValueType>* const b,
                  matrix::BatchDense<ValueType>* const c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
    for (size_type batch = 0; batch < c->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        matvec_kernel(a_b, b_b, c_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::BatchDense<ValueType>* const alpha,
           const matrix::BatchDense<ValueType>* const a,
           const matrix::BatchDense<ValueType>* const b,
           const matrix::BatchDense<ValueType>* const beta,
           matrix::BatchDense<ValueType>* const c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
    for (size_type batch = 0; batch < c->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto beta_b = gko::batch::batch_entry(beta_ub, batch);
        advanced_matvec_kernel(alpha_b.values[0], a_b, b_b, beta_b.values[0],
                               c_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::BatchDense<ValueType>* alpha,
           matrix::BatchDense<ValueType>* x)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        scale(alpha_b, x_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::BatchDense<ValueType>* alpha,
                const matrix::BatchDense<ValueType>* x,
                matrix::BatchDense<ValueType>* y)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        add_scaled(alpha_b, x_b, y_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scale(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::BatchDense<ValueType>* const alpha,
               const matrix::BatchDense<ValueType>* const x,
               const matrix::BatchDense<ValueType>* const beta,
               matrix::BatchDense<ValueType>* const y)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
    for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto beta_b = gko::batch::batch_entry(beta_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        add_scale(alpha_b, x_b, beta_b, y_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALE_KERNEL);


template <typename ValueType>
void convergence_add_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                            const matrix::BatchDense<ValueType>* alpha,
                            const matrix::BatchDense<ValueType>* x,
                            matrix::BatchDense<ValueType>* y,
                            const uint32& converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
    for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        add_scaled(alpha_b, x_b, y_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const ReferenceExecutor> exec,
                     const matrix::BatchDense<ValueType>* alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::BatchDense<ValueType>* y) GKO_NOT_IMPLEMENTED;
// {
// for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
//     const auto diag_values = x->get_const_values();
//     for (size_type i = 0; i < x->get_size().at(batch)[0]; i++) {
//         y->at(batch,i, i) += alpha->at(batch,0, 0) * diag_values[i];
//     }
// }
// }

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const ReferenceExecutor> exec,
                 const matrix::BatchDense<ValueType>* x,
                 const matrix::BatchDense<ValueType>* y,
                 matrix::BatchDense<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        compute_dot_product(x_b, y_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void convergence_compute_dot(std::shared_ptr<const ReferenceExecutor> exec,
                             const matrix::BatchDense<ValueType>* x,
                             const matrix::BatchDense<ValueType>* y,
                             matrix::BatchDense<ValueType>* result,
                             const uint32& converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        compute_dot_product(x_b, y_b, res_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::BatchDense<ValueType>* x,
                   matrix::BatchDense<remove_complex<ValueType>>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        compute_norm2(x_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void convergence_compute_norm2(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::BatchDense<ValueType>* x,
    matrix::BatchDense<remove_complex<ValueType>>* result,
    const uint32& converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto res_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        compute_norm2(x_b, res_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType>* source,
                          matrix::BatchCsr<ValueType, IndexType>* result)
{
    GKO_ASSERT(source->get_size().stores_equal_sizes() == true);
    auto num_rows = result->get_size().at(0)[0];
    auto num_cols = result->get_size().at(0)[1];
    auto num_batch_entries = result->get_num_batch_entries();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    size_type cur_ptr = 0;
    row_ptrs[0] = cur_ptr;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(0, row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = col;
                ++cur_ptr;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }

    cur_ptr = 0;
    for (size_type batch = 0; batch < num_batch_entries; ++batch) {
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                auto val = source->at(batch, row, col);
                if (val != zero<ValueType>()) {
                    values[cur_ptr] = val;
                    ++cur_ptr;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_DENSE_CONVERT_TO_BATCH_CSR_KERNEL);


template <typename ValueType>
void count_nonzeros(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::BatchDense<ValueType>* source,
                    size_type* result)
{
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        auto num_nonzeros = 0;

        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros +=
                    (source->at(batch, row, col) != zero<ValueType>());
            }
        }
        result[batch] = num_nonzeros;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                               const matrix::BatchDense<ValueType>* source,
                               size_type* result)
{
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        size_type num_stored_elements_per_row = 0;
        size_type num_nonzeros = 0;
        for (size_type row = 0; row < num_rows; ++row) {
            num_nonzeros = 0;
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros +=
                    (source->at(batch, row, col) != zero<ValueType>());
            }
            num_stored_elements_per_row =
                std::max(num_nonzeros, num_stored_elements_per_row);
        }
        result[batch] = num_stored_elements_per_row;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_nonzeros_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                                const matrix::BatchDense<ValueType>* source,
                                array<size_type>* result)
{
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        auto row_nnz_val = result->get_data();
        size_type offset = 0;
        for (size_type row = 0; row < num_rows; ++row) {
            size_type num_nonzeros = 0;
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros +=
                    (source->at(batch, row, col) != zero<ValueType>());
            }
            row_nnz_val[offset + row] = num_nonzeros;
            ++offset;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const ReferenceExecutor> exec,
                          const matrix::BatchDense<ValueType>* const source,
                          size_type* const result,
                          const size_type* const stride_factor,
                          const size_type* const slice_size)
{
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        auto slice_num = ceildiv(num_rows, slice_size[batch]);
        auto total_cols = 0;
        auto temp = 0, slice_temp = 0;
        for (size_type slice = 0; slice < slice_num; slice++) {
            slice_temp = 0;
            for (size_type row = 0; row < slice_size[batch] &&
                                    row + slice * slice_size[batch] < num_rows;
                 row++) {
                temp = 0;
                for (size_type col = 0; col < num_cols; col++) {
                    temp += (source->at(batch, row + slice * slice_size[batch],
                                        col) != zero<ValueType>());
                }
                slice_temp = (slice_temp < temp) ? temp : slice_temp;
            }
            slice_temp = ceildiv(slice_temp, stride_factor[batch]) *
                         stride_factor[batch];
            total_cols += slice_temp;
        }
        result[batch] = total_cols;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::BatchDense<ValueType>* const orig,
               matrix::BatchDense<ValueType>* const trans)
{
    for (size_type batch = 0; batch < orig->get_num_batch_entries(); ++batch) {
        for (size_type i = 0; i < orig->get_size().at(batch)[0]; ++i) {
            for (size_type j = 0; j < orig->get_size().at(batch)[1]; ++j) {
                trans->at(batch, j, i) = orig->at(batch, i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::BatchDense<ValueType>* orig,
                    matrix::BatchDense<ValueType>* trans)
{
    for (size_type batch = 0; batch < orig->get_num_batch_entries(); ++batch) {
        for (size_type i = 0; i < orig->get_size().at(batch)[0]; ++i) {
            for (size_type j = 0; j < orig->get_size().at(batch)[1]; ++j) {
                trans->at(batch, j, i) = conj(orig->at(batch, i, j));
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::BatchDense<ValueType>* x,
          matrix::BatchDense<ValueType>* result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto result_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto result_b = gko::batch::batch_entry(result_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        copy(x_b, result_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COPY_KERNEL);


template <typename ValueType>
void convergence_copy(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::BatchDense<ValueType>* x,
                      matrix::BatchDense<ValueType>* result,
                      const uint32& converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto result_ub = host::get_batch_struct(result);
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto result_b = gko::batch::batch_entry(result_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        copy(x_b, result_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL);


template <typename ValueType>
void batch_scale(std::shared_ptr<const ReferenceExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left,
                 const matrix::BatchDiagonal<ValueType>* const rght,
                 matrix::BatchDense<ValueType>* const vecs)
{
    const auto left_vals = left->get_const_values();
    const auto rght_vals = rght->get_const_values();
    const auto v_vals = vecs->get_values();
    const auto nrows = static_cast<int>(vecs->get_size().at(0)[0]);
    const auto ncols = static_cast<int>(vecs->get_size().at(0)[1]);
    const auto vstride = vecs->get_stride().at(0);
    for (size_type batch = 0; batch < vecs->get_num_batch_entries(); ++batch) {
        const auto left_b =
            gko::batch::batch_entry_ptr(left_vals, 1, nrows, batch);
        const auto rght_b =
            gko::batch::batch_entry_ptr(rght_vals, 1, ncols, batch);
        const auto v_b =
            gko::batch::batch_entry_ptr(v_vals, vstride, nrows, batch);
        batch_scale(nrows, ncols, vstride, left_b, rght_b, v_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


template <typename ValueType>
void add_scaled_identity(std::shared_ptr<const ReferenceExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchDense<ValueType>* const mtx)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto mtx_ub = host::get_batch_struct(mtx);
    for (size_type batch = 0; batch < mtx->get_num_batch_entries(); ++batch) {
        auto a_b = gko::batch::batch_entry(a_ub, batch);
        auto b_b = gko::batch::batch_entry(b_ub, batch);
        auto mtx_b = gko::batch::batch_entry(mtx_ub, batch);
        add_scaled_identity(a_b.values[0], b_b.values[0], mtx_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
