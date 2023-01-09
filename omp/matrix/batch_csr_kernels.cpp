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

#include "core/matrix/batch_csr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "reference/matrix/batch_struct.hpp"

namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {

#include "reference/matrix/batch_csr_kernels.hpp.inc"

template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType>* a,
          const matrix::BatchDense<ValueType>* b,
          matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        matvec_kernel(a_b, b_b, c_b);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::BatchDense<ValueType>* alpha,
                   const matrix::BatchCsr<ValueType, IndexType>* a,
                   const matrix::BatchDense<ValueType>* b,
                   const matrix::BatchDense<ValueType>* beta,
                   matrix::BatchDense<ValueType>* c)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto c_ub = host::get_batch_struct(c);
    const auto alpha_ub = host::get_batch_struct(alpha);
    const auto beta_ub = host::get_batch_struct(beta);
#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto c_b = gko::batch::batch_entry(c_ub, batch);
        const auto alpha_b = gko::batch::batch_entry(alpha_ub, batch);
        const auto beta_b = gko::batch::batch_entry(beta_ub, batch);
        advanced_matvec_kernel(alpha_b.values[0], a_b, b_b, beta_b.values[0],
                               c_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void batch_scale(std::shared_ptr<const OmpExecutor> exec,
                 const matrix::BatchDiagonal<ValueType>* const left_scale,
                 const matrix::BatchDiagonal<ValueType>* const right_scale,
                 matrix::BatchCsr<ValueType, IndexType>* const mat)
{
    if (!left_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    if (!right_scale->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;

    const size_type nbatches = mat->get_num_batch_entries();
    const auto a_ub = host::get_batch_struct(mat);
    const auto nrows = static_cast<int>(mat->get_size().at(0)[0]);
    const auto ncols = static_cast<int>(mat->get_size().at(0)[1]);
    const auto left_vals = left_scale->get_const_values();
    const auto rght_vals = right_scale->get_const_values();

#pragma omp parallel for
    for (size_type ibatch = 0; ibatch < nbatches; ibatch++) {
        auto a_b = gko::batch::batch_entry(a_ub, ibatch);
        const auto left_b =
            gko::batch::batch_entry_ptr(left_vals, 1, nrows, ibatch);
        const auto right_b =
            gko::batch::batch_entry_ptr(rght_vals, 1, ncols, ibatch);
        batch_scale(left_b, right_b, a_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SCALE);


template <typename ValueType, typename IndexType>
void pre_diag_transform_system(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchDiagonal<ValueType>* const left_op,
    const matrix::BatchDiagonal<ValueType>* const right_op,
    matrix::BatchCsr<ValueType, IndexType>* const a,
    matrix::BatchDense<ValueType>* const b)
{
    const size_type nbatch = a->get_num_batch_entries();
    const int nrows = static_cast<int>(a->get_size().at()[0]);
    const int ncols = static_cast<int>(a->get_size().at()[1]);
    const size_type nnz = a->get_num_stored_elements() / nbatch;
    const int nrhs = static_cast<int>(b->get_size().at()[1]);
    const size_type b_stride = b->get_stride().at();
#pragma omp parallel for
    for (size_type ib = 0; ib < nbatch; ib++) {
        pre_diag_scale_system(
            ib, nnz, nrows, ncols, a->get_values(), a->get_const_col_idxs(),
            a->get_const_row_ptrs(), nrhs, b_stride, b->get_values(),
            left_op->get_const_values(), right_op->get_const_values());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_PRE_DIAG_TRANSFORM_SYSTEM);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const OmpExecutor> exec,
                              const IndexType* ptrs, size_type num_rows,
                              IndexType* idxs) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::BatchCsr<ValueType, IndexType>* source,
                      matrix::BatchDense<ValueType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_batch_csr_to_csc(
    size_type num_rows, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* batch_csr_vals, IndexType* row_idxs, IndexType* col_ptrs,
    ValueType* csc_vals, UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const OmpExecutor> exec,
                             matrix::BatchCsr<ValueType, IndexType>* trans,
                             const matrix::BatchCsr<ValueType, IndexType>* orig,
                             UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType>* orig,
               matrix::BatchCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType>* orig,
                    matrix::BatchCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::BatchCsr<ValueType, IndexType>* source,
                          size_type* result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* source,
    size_type* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* source,
    array<size_type>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void convert_to_batch_dense(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const src,
    matrix::BatchDense<ValueType>* const dest)
{
    const size_type nbatches = src->get_num_batch_entries();
    const int nrows = src->get_size().at()[0];
    const int ncols = src->get_size().at()[1];
    const int nnz = src->get_const_row_ptrs()[nrows];
    const size_type dstride = dest->get_stride().at();
#pragma omp parallel for
    for (size_type ibatch = 0; ibatch < nbatches; ibatch++) {
        convert_csr_to_dense(nrows, ncols, src->get_const_row_ptrs(),
                             src->get_const_col_idxs(),
                             src->get_const_values() + ibatch * nnz, dstride,
                             dest->get_values() + ibatch * dstride * nrows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_BATCH_DENSE);


template <typename ValueType, typename IndexType>
void check_diagonal_entries_exist(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const mtx,
    bool& has_all_diags)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto nmin = static_cast<int>(
        std::min(mtx->get_size().at(0)[0], mtx->get_size().at(0)[1]));
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    check_diagonal_entries_exist(nmin, row_ptrs, col_idxs, has_all_diags);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CHECK_DIAGONAL_ENTRIES_EXIST);


template <typename ValueType, typename IndexType>
void add_scaled_identity(std::shared_ptr<const OmpExecutor> exec,
                         const matrix::BatchDense<ValueType>* const a,
                         const matrix::BatchDense<ValueType>* const b,
                         matrix::BatchCsr<ValueType, IndexType>* const mtx)
{
    if (!mtx->get_size().stores_equal_sizes()) GKO_NOT_IMPLEMENTED;
    const auto batch_size = mtx->get_num_batch_entries();
    const auto nrows = static_cast<int>(mtx->get_size().at(0)[0]);
    const IndexType nnz = mtx->get_const_row_ptrs()[nrows];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto col_idxs = mtx->get_const_col_idxs();
    const auto a_stride = a->get_stride().at();
    const auto b_stride = b->get_stride().at();
#pragma omp parallel for
    for (size_type batch = 0; batch < batch_size; batch++) {
        const auto values = mtx->get_values() + batch * nnz;
        const auto aptr =
            batch::batch_entry_ptr(a->get_const_values(), a_stride, 1, batch);
        const auto bptr =
            batch::batch_entry_ptr(b->get_const_values(), b_stride, 1, batch);
        add_scaled_identity(nrows, row_ptrs, col_idxs, values, aptr[0],
                            bptr[0]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace batch_csr
}  // namespace omp
}  // namespace kernels
}  // namespace gko
