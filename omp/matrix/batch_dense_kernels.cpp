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

#include "core/matrix/batch_dense_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "reference/matrix/batch_dense_kernels.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The BatchDense matrix format namespace.
 * @ref BatchDense
 * @ingroup batch_dense
 */
namespace batch_dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::BatchDense<ValueType> *const a,
                  const matrix::BatchDense<ValueType> *const b,
                  matrix::BatchDense<ValueType> *const c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
#pragma omp parallel for
    for (size_type batch = 0; batch < c->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        gko::kernels::reference::batch_dense::simple_apply(a_b, b_b, c_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const OmpExecutor> exec,
           const matrix::BatchDense<ValueType> *const alpha,
           const matrix::BatchDense<ValueType> *const a,
           const matrix::BatchDense<ValueType> *const b,
           const matrix::BatchDense<ValueType> *const beta,
           matrix::BatchDense<ValueType> *const c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
#pragma omp parallel for
    for (size_type batch = 0; batch < c->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto beta_b = gko::batch::batch_entry(beta_ub, batch);
        gko::kernels::reference::batch_dense::apply(alpha_b.values[0], a_b, b_b,
                                                    beta_b.values[0], c_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_APPLY_KERNEL);


template <typename ValueType>
void scale(std::shared_ptr<const OmpExecutor> exec,
           const matrix::BatchDense<ValueType> *const alpha,
           matrix::BatchDense<ValueType> *const x)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto alpha_ub = host::get_batch_struct(alpha);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        gko::kernels::reference::batch_dense::scale(alpha_b, x_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_SCALE_KERNEL);


template <typename ValueType>
void convergence_scale(std::shared_ptr<const OmpExecutor> exec,
                       const matrix::BatchDense<ValueType> *const alpha,
                       matrix::BatchDense<ValueType> *const x,
                       const uint32 &converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto alpha_ub = host::get_batch_struct(alpha);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        gko::kernels::reference::batch_dense::scale(alpha_b, x_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_SCALE_KERNEL);


template <typename ValueType>
void add_scaled(std::shared_ptr<const OmpExecutor> exec,
                const matrix::BatchDense<ValueType> *const alpha,
                const matrix::BatchDense<ValueType> *const x,
                matrix::BatchDense<ValueType> *const y)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
#pragma omp parallel for
    for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        gko::kernels::reference::batch_dense::add_scaled(alpha_b, x_b, y_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType>
void convergence_add_scaled(std::shared_ptr<const OmpExecutor> exec,
                            const matrix::BatchDense<ValueType> *const alpha,
                            const matrix::BatchDense<ValueType> *const x,
                            matrix::BatchDense<ValueType> *const y,
                            const uint32 &converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto alpha_ub = host::get_batch_struct(alpha);
#pragma omp parallel for
    for (size_type batch = 0; batch < y->get_num_batch_entries(); ++batch) {
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        gko::kernels::reference::batch_dense::add_scaled(alpha_b, x_b, y_b,
                                                         converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_ADD_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const OmpExecutor>,
                     const matrix::BatchDense<ValueType> *,
                     const matrix::Diagonal<ValueType> *,
                     matrix::BatchDense<ValueType> *) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const OmpExecutor> exec,
                 const matrix::BatchDense<ValueType> *const x,
                 const matrix::BatchDense<ValueType> *const y,
                 matrix::BatchDense<ValueType> *const result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
#pragma omp parallel for
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        gko::kernels::reference::batch_dense::compute_dot_product(x_b, y_b,
                                                                  res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void convergence_compute_dot(std::shared_ptr<const OmpExecutor> exec,
                             const matrix::BatchDense<ValueType> *const x,
                             const matrix::BatchDense<ValueType> *const y,
                             matrix::BatchDense<ValueType> *const result,
                             const uint32 &converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto y_ub = host::get_batch_struct(y);
    const auto res_ub = host::get_batch_struct(result);
#pragma omp parallel for
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        const auto y_b = gko::batch::batch_entry(y_ub, batch);
        gko::kernels::reference::batch_dense::compute_dot_product(
            x_b, y_b, res_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::BatchDense<ValueType> *const x,
                   matrix::BatchDense<remove_complex<ValueType>> *const result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto res_ub = host::get_batch_struct(result);
#pragma omp parallel for
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        gko::kernels::reference::batch_dense::compute_norm2(x_b, res_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void convergence_compute_norm2(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchDense<ValueType> *const x,
    matrix::BatchDense<remove_complex<ValueType>> *const result,
    const uint32 &converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto res_ub = host::get_batch_struct(result);
#pragma omp parallel for
    for (size_type batch = 0; batch < result->get_num_batch_entries();
         ++batch) {
        const auto res_b = gko::batch::batch_entry(res_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        gko::kernels::reference::batch_dense::compute_norm2(x_b, res_b,
                                                            converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COMPUTE_NORM2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_batch_csr(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::BatchDense<ValueType> *const source,
                          matrix::BatchCsr<ValueType, IndexType> *const result)
{
    GKO_ASSERT(source->get_size().stores_equal_sizes() == true);
    auto num_rows = result->get_size().at(0)[0];
    auto num_cols = result->get_size().at(0)[1];
    auto num_batches = result->get_num_batch_entries();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();


#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType row_nnz{};
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(0, row, col);
            row_nnz += static_cast<IndexType>(val != zero<ValueType>());
        }
        row_ptrs[row] = row_nnz;
    }

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto cur_ptr = row_ptrs[row];
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(0, row, col);
            if (val != zero<ValueType>()) {
                col_idxs[cur_ptr] = static_cast<IndexType>(col);
                ++cur_ptr;
            }
        }
    }

#pragma omp parallel for
    for (size_type batch = 0; batch < num_batches; ++batch) {
        size_type cur_ptr =
            batch * row_ptrs[num_rows];  // as row_ptrs[num_rows] is the num of
                                         // non zero elements in the matrix
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
void count_nonzeros(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::BatchDense<ValueType> *const source,
                    size_type *const result)
{
#pragma omp parallel for
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        size_type num_nonzeros = 0;

        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros += static_cast<size_type>(
                    source->at(batch, row, col) != zero<ValueType>());
            }
        }
        result[batch] = num_nonzeros;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_COUNT_NONZEROS_KERNEL);


template <typename ValueType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const OmpExecutor>,
    const matrix::BatchDense<ValueType> *const source, size_type *const result)
{
#pragma omp parallel for
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        size_type num_stored_elements_per_row = 0;
        size_type num_nonzeros = 0;

        for (size_type row = 0; row < num_rows; ++row) {
            num_nonzeros = 0;
            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros += static_cast<size_type>(
                    source->at(batch, row, col) != zero<ValueType>());
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
void calculate_nonzeros_per_row(
    std::shared_ptr<const OmpExecutor>,
    const matrix::BatchDense<ValueType> *const source,
    Array<size_type> *const result)
{
    size_type cumul_prev_rows = 0;
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        auto row_nnz_val = result->get_data() + cumul_prev_rows;

#pragma omp parallel for reduction(+ : cumul_prev_rows)
        for (size_type row = 0; row < num_rows; ++row) {
            size_type num_nonzeros = 0;

            for (size_type col = 0; col < num_cols; ++col) {
                num_nonzeros += static_cast<size_type>(
                    source->at(batch, row, col) != zero<ValueType>());
            }
            row_nnz_val[row] = num_nonzeros;
            ++cumul_prev_rows;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType>
void calculate_total_cols(std::shared_ptr<const OmpExecutor>,
                          const matrix::BatchDense<ValueType> *const source,
                          size_type *const result,
                          const size_type *const stride_factor,
                          const size_type *const slice_size)
{
#pragma omp parallel for
    for (size_type batch = 0; batch < source->get_num_batch_entries();
         ++batch) {
        auto num_rows = source->get_size().at(batch)[0];
        auto num_cols = source->get_size().at(batch)[1];
        auto slice_num = ceildiv(num_rows, slice_size[batch]);
        size_type total_cols = 0;
        size_type temp = 0;
        size_type slice_temp = 0;

        for (size_type slice = 0; slice < slice_num; slice++) {
            slice_temp = 0;
            for (size_type row = 0; row < slice_size[batch] &&
                                    row + slice * slice_size[batch] < num_rows;
                 row++) {
                temp = 0;
                for (size_type col = 0; col < num_cols; col++) {
                    temp += static_cast<size_type>(
                        source->at(batch, row + slice * slice_size[batch],
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
void transpose(std::shared_ptr<const OmpExecutor>,
               const matrix::BatchDense<ValueType> *const orig,
               matrix::BatchDense<ValueType> *const trans)
{
#pragma omp parallel for
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
void conj_transpose(std::shared_ptr<const OmpExecutor>,
                    const matrix::BatchDense<ValueType> *const orig,
                    matrix::BatchDense<ValueType> *const trans)
{
#pragma omp parallel for
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
          const matrix::BatchDense<ValueType> *x,
          matrix::BatchDense<ValueType> *result)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto result_ub = host::get_batch_struct(result);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto result_b = gko::batch::batch_entry(result_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        gko::kernels::reference::batch_dense::copy(x_b, result_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_COPY_KERNEL);


template <typename ValueType>
void convergence_copy(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::BatchDense<ValueType> *x,
                      matrix::BatchDense<ValueType> *result,
                      const uint32 &converged)
{
    const auto x_ub = host::get_batch_struct(x);
    const auto result_ub = host::get_batch_struct(result);
#pragma omp parallel for
    for (size_type batch = 0; batch < x->get_num_batch_entries(); ++batch) {
        const auto result_b = gko::batch::batch_entry(result_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);
        gko::kernels::reference::batch_dense::copy(x_b, result_b, converged);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_DENSE_CONVERGENCE_COPY_KERNEL);


template <typename ValueType>
void batch_scale(std::shared_ptr<const OmpExecutor> exec,
                 const matrix::BatchDense<ValueType> *const scale_vec,
                 matrix::BatchDense<ValueType> *const vecs)
{
    const auto scale_ub = host::get_batch_struct(scale_vec);
    const auto v_ub = host::get_batch_struct(vecs);
#pragma omp parallel for
    for (size_type batch = 0; batch < vecs->get_num_batch_entries(); ++batch) {
        const auto sc_b = gko::batch::batch_entry(scale_ub, batch);
        const auto v_b = gko::batch::batch_entry(v_ub, batch);
        gko::kernels::reference::batch_dense::batch_scale(sc_b, v_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_DENSE_BATCH_SCALE_KERNEL);


}  // namespace batch_dense
}  // namespace omp
}  // namespace kernels
}  // namespace gko
