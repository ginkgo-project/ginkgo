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

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"
#include "omp/components/fbcsr_spgeam.hpp"
#include "omp/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup fbcsr
 */
namespace fbcsr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Fbcsr<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b,
          matrix::Dense<ValueType> *c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Fbcsr<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spgemm_insert_row(unordered_set<IndexType> &cols,
                       const matrix::Fbcsr<ValueType, IndexType> *c,
                       size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto row_ptrs = c->get_const_row_ptrs();
//    auto col_idxs = c->get_const_col_idxs();
//    cols.insert(col_idxs + row_ptrs[row], col_idxs + row_ptrs[row + 1]);
//}


template <typename ValueType, typename IndexType>
void spgemm_insert_row2(unordered_set<IndexType> &cols,
                        const matrix::Fbcsr<ValueType, IndexType> *a,
                        const matrix::Fbcsr<ValueType, IndexType> *b,
                        size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto a_row_ptrs = a->get_const_row_ptrs();
//    auto a_col_idxs = a->get_const_col_idxs();
//    auto b_row_ptrs = b->get_const_row_ptrs();
//    auto b_col_idxs = b->get_const_col_idxs();
//    for (size_type a_nz = a_row_ptrs[row];
//         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
//        auto a_col = a_col_idxs[a_nz];
//        auto b_row = a_col;
//        cols.insert(b_col_idxs + b_row_ptrs[b_row],
//                    b_col_idxs + b_row_ptrs[b_row + 1]);
//    }
//}


template <typename IndexType>
void convert_row_ptrs_to_idxs(std::shared_ptr<const OmpExecutor> exec,
                              const IndexType *ptrs, size_type num_rows,
                              IndexType *idxs) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    convert_ptrs_to_idxs(ptrs, num_rows, idxs);
//}


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *const source,
                    matrix::Csr<ValueType, IndexType> *const result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator>
inline void convert_fbcsr_to_csc(size_type num_rows, const IndexType *row_ptrs,
                                 const IndexType *col_idxs,
                                 const ValueType *fbcsr_vals,
                                 IndexType *row_idxs, IndexType *col_ptrs,
                                 ValueType *csc_vals,
                                 UnaryOperator op) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(std::shared_ptr<const OmpExecutor> exec,
                             matrix::Fbcsr<ValueType, IndexType> *trans,
                             const matrix::Fbcsr<ValueType, IndexType> *orig,
                             UnaryOperator op) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
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
//    convert_fbcsr_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs,
//    orig_vals,
//                       trans_col_idxs, trans_row_ptrs + 1, trans_vals, op);
//}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType> *orig,
               matrix::Fbcsr<ValueType, IndexType> *trans) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    transpose_and_transform(exec, trans, orig,
//                            [](const ValueType x) { return x; });
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *orig,
                    matrix::Fbcsr<ValueType, IndexType> *trans)
    GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    transpose_and_transform(exec, trans, orig,
//                            [](const ValueType x) { return conj(x); });
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *source,
    size_type *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *source,
    Array<size_type> *result) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    const auto row_ptrs = source->get_const_row_ptrs();
//    auto row_nnz_val = result->get_data();
//
//#pragma omp parallel for
//    for (size_type i = 0; i < result->get_num_elems(); i++) {
//        row_nnz_val[i] = row_ptrs[i + 1] - row_ptrs[i];
//    }
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *to_check,
    bool *is_sorted) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
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

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *orig,
                      matrix::Diagonal<ValueType> *diag) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL);


}  // namespace fbcsr
}  // namespace omp
}  // namespace kernels
}  // namespace gko
