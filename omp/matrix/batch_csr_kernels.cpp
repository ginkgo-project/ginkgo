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

#include "core/matrix/batch_csr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"
#include "omp/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup batch_csr
 */
namespace batch_csr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::BatchCsr<ValueType, IndexType> *a,
          const matrix::BatchDense<ValueType> *b,
          matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto row_ptrs = a->get_const_row_ptrs();
//    auto col_idxs = a->get_const_col_idxs();
//    auto vals = a->get_const_values();
//
//#pragma omp parallel for
//    for (size_type row = 0; row < a->get_size()[0]; ++row) {
//        for (size_type j = 0; j < c->get_size()[1]; ++j) {
//            c->at(row, j) = zero<ValueType>();
//        }
//        for (size_type k = row_ptrs[row];
//             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
//            auto val = vals[k];
//            auto col = col_idxs[k];
//            for (size_type j = 0; j < c->get_size()[1]; ++j) {
//                c->at(row, j) += val * b->at(col, j);
//            }
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::BatchDense<ValueType> *alpha,
                   const matrix::BatchCsr<ValueType, IndexType> *a,
                   const matrix::BatchDense<ValueType> *b,
                   const matrix::BatchDense<ValueType> *beta,
                   matrix::BatchDense<ValueType> *c) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto row_ptrs = a->get_const_row_ptrs();
//    auto col_idxs = a->get_const_col_idxs();
//    auto vals = a->get_const_values();
//    auto valpha = alpha->at(0, 0);
//    auto vbeta = beta->at(0, 0);
//
//#pragma omp parallel for
//    for (size_type row = 0; row < a->get_size()[0]; ++row) {
//        for (size_type j = 0; j < c->get_size()[1]; ++j) {
//            c->at(row, j) *= vbeta;
//        }
//        for (size_type k = row_ptrs[row];
//             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
//            auto val = vals[k];
//            auto col = col_idxs[k];
//            for (size_type j = 0; j < c->get_size()[1]; ++j) {
//                c->at(row, j) += valpha * val * b->at(col, j);
//            }
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const OmpExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    convert_ptrs_to_idxs(ptrs, num_rows, idxs);
//}


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::BatchCsr<ValueType, IndexType> *source,
                      matrix::BatchDense<ValueType> *result)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto num_rows = source->get_size()[0];
//    auto num_cols = source->get_size()[1];
//    auto row_ptrs = source->get_const_row_ptrs();
//    auto col_idxs = source->get_const_col_idxs();
//    auto vals = source->get_const_values();
//
//#pragma omp parallel for
//    for (size_type row = 0; row < num_rows; ++row) {
//        for (size_type col = 0; col < num_cols; ++col) {
//            result->at(row, col) = zero<ValueType>();
//        }
//        for (size_type i = row_ptrs[row];
//             i < static_cast<size_type>(row_ptrs[row + 1]); ++i) {
//            result->at(row, col_idxs[i]) = vals[i];
//        }
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_batch_csr_to_csc(
    size_type num_rows, const IndexType *row_ptrs, const IndexType *col_idxs,
    const ValueType *batch_csr_vals, IndexType *row_idxs, IndexType *col_ptrs,
    ValueType *csc_vals, UnaryOperator op) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    for (size_type row = 0; row < num_rows; ++row) {
//        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
//            const auto dest_idx = col_ptrs[col_idxs[i]]++;
//            row_idxs[dest_idx] = row;
//            csc_vals[dest_idx] = op(batch_csr_vals[i]);
//        }
//    }
//}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const OmpExecutor> exec,
                             matrix::BatchCsr<ValueType, IndexType> *trans,
                             const matrix::BatchCsr<ValueType, IndexType> *orig,
                             UnaryOperator op) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto trans_row_ptrs = trans->get_row_ptrs();
//    auto orig_row_ptrs = orig->get_const_row_ptrs();
//    auto trans_col_idxs = trans->get_col_idxs();
//    auto orig_col_idxs = orig->get_const_col_idxs();
//    auto trans_vals = trans->get_values();
//    auto orig_vals = orig->get_const_values();
//
//    auto orig_num_cols = orig->get_size()[1];
//    auto orig_num_rows = orig->get_size()[0];
//    auto orig_nnz = orig_row_ptrs[orig_num_rows];
//
//    trans_row_ptrs[0] = 0;
//    convert_unsorted_idxs_to_ptrs(orig_col_idxs, orig_nnz, trans_row_ptrs + 1,
//                                  orig_num_cols);
//
//    convert_batch_csr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs,
//    orig_vals,
//                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
//}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::BatchCsr<ValueType, IndexType> *orig,
               matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    transpose_and_transform(exec, trans, orig,
//                            [](const ValueType x) { return x; });
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType> *orig,
                    matrix::BatchCsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    transpose_and_transform(exec, trans, orig,
//                            [](const ValueType x) { return conj(x); });
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_total_cols(std::shared_ptr<const OmpExecutor> exec,
                          const matrix::BatchCsr<ValueType, IndexType> *source,
                          size_type *result, size_type stride_factor,
                          size_type slice_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_TOTAL_COLS_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    size_type *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *source,
    Array<size_type> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto row_ptrs = source->get_const_row_ptrs();
//    auto row_nnz_val = result->get_data();
//
//#pragma omp parallel for
//    for (size_type i = 0; i < result->get_num_elems(); i++) {
//        row_nnz_val[i] = row_ptrs[i + 1] - row_ptrs[i];
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::BatchCsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    auto values = to_sort->get_values();
//    auto row_ptrs = to_sort->get_row_ptrs();
//    auto col_idxs = to_sort->get_col_idxs();
//    const auto number_rows = to_sort->get_size()[0];
//#pragma omp parallel for
//    for (size_type i = 0; i < number_rows; ++i) {
//        auto start_row_idx = row_ptrs[i];
//        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
//        auto helper = detail::IteratorFactory<IndexType, ValueType>(
//            col_idxs + start_row_idx, values + start_row_idx, row_nnz);
//        std::sort(helper.begin(), helper.end());
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType> *to_check,
    bool *is_sorted) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_csr): change the code imported from matrix/csr if needed
//    const auto row_ptrs = to_check->get_const_row_ptrs();
//    const auto col_idxs = to_check->get_const_col_idxs();
//    const auto size = to_check->get_size();
//    bool local_is_sorted = true;
//#pragma omp parallel for reduction(&& : local_is_sorted)
//    for (size_type i = 0; i < size[0]; ++i) {
//        // Skip comparison if any thread detects that it is not sorted
//        if (local_is_sorted) {
//            for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
//                if (col_idxs[idx - 1] > col_idxs[idx]) {
//                    local_is_sorted = false;
//                    break;
//                }
//            }
//        }
//    }
//    *is_sorted = local_is_sorted;
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace batch_csr
}  // namespace omp
}  // namespace kernels
}  // namespace gko
