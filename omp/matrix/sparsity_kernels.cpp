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

#include "core/matrix/sparsity_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/iterator_factory.hpp"
#include "omp/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Sparsity pattern format namespace.
 *
 * @ingroup sparsity
 */
namespace sparsity {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Sparsity<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) = zero<ValueType>();
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = one<ValueType>();
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SPARSITY_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Sparsity<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            c->at(row, j) *= vbeta;
        }
        for (size_type k = row_ptrs[row];
             k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
            auto val = one<ValueType>();
            auto col = col_idxs[k];
            for (size_type j = 0; j < c->get_size()[1]; ++j) {
                c->at(row, j) += valpha * val * b->at(col, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void count_num_diagonal_elements(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Sparsity<ValueType, IndexType> *matrix,
    size_type &num_diagonal_elements)
{
    auto num_rows = matrix->get_size()[0];
    auto row_ptrs = matrix->get_const_row_ptrs();
    auto col_idxs = matrix->get_const_col_idxs();
    size_type num_diag = 0;
    for (auto i = 0; i < num_rows; ++i) {
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            if (col_idxs[j] == i) {
                num_diag++;
            }
        }
    }
    num_diagonal_elements = num_diag;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void remove_diagonal_elements(std::shared_ptr<const OmpExecutor> exec,
                              matrix::Sparsity<ValueType, IndexType> *matrix,
                              const IndexType *row_ptrs,
                              const IndexType *col_idxs)
{
    auto num_rows = matrix->get_size()[0];
    auto adj_ptrs = matrix->get_row_ptrs();
    auto adj_idxs = matrix->get_col_idxs();
    for (auto i = 0; i <= num_rows; ++i) {
        adj_ptrs[i] = row_ptrs[i] - i;
    }
    std::vector<IndexType> temp_idxs;
    for (auto i = 0; i < num_rows; ++i) {
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            if (col_idxs[j] != i) {
                temp_idxs.push_back(col_idxs[j]);
            }
        }
    }
#pragma omp parallel for
    for (auto i = 0; i < temp_idxs.size(); ++i) {
        adj_idxs[i] = temp_idxs[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_REMOVE_DIAGONAL_ELEMENTS_KERNEL);


template <typename IndexType>
inline void convert_sparsity_to_csc(size_type num_rows,
                                    const IndexType *row_ptrs,
                                    const IndexType *col_idxs,
                                    IndexType *row_idxs, IndexType *col_ptrs)
{
    for (size_type row = 0; row < num_rows; ++row) {
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = row;
        }
    }
}


template <typename ValueType, typename IndexType>
void transpose_and_transform(std::shared_ptr<const OmpExecutor> exec,
                             matrix::Sparsity<ValueType, IndexType> *trans,
                             const matrix::Sparsity<ValueType, IndexType> *orig)
{
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();

    auto orig_num_cols = orig->get_size()[1];
    auto orig_num_rows = orig->get_size()[0];
    auto orig_nnz = orig_row_ptrs[orig_num_rows];

    trans_row_ptrs[0] = 0;
    convert_unsorted_idxs_to_ptrs(orig_col_idxs, orig_nnz, trans_row_ptrs + 1,
                                  orig_num_cols);

    convert_sparsity_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs,
                            trans_col_idxs, trans_row_ptrs + 1);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const OmpExecutor> exec,
               matrix::Sparsity<ValueType, IndexType> *trans,
               const matrix::Sparsity<ValueType, IndexType> *orig)
{
    transpose_and_transform(exec, trans, orig);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const OmpExecutor> exec,
                          matrix::Sparsity<ValueType, IndexType> *to_sort)
{
    auto row_ptrs = to_sort->get_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    const auto number_rows = to_sort->get_size()[0];
#pragma omp parallel for
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        std::sort(col_idxs + start_row_idx, col_idxs + start_row_idx + row_nnz);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Sparsity<ValueType, IndexType> *to_check, bool *is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    bool local_is_sorted = true;
#pragma omp parallel for shared(local_is_sorted)
    for (size_type i = 0; i < size[0]; ++i) {
#pragma omp flush(local_is_sorted)
        // Skip comparison if any thread detects that it is not sorted
        if (local_is_sorted) {
            for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
                if (col_idxs[idx - 1] > col_idxs[idx]) {
                    local_is_sorted = false;
                    break;
                }
            }
        }
    }
    *is_sorted = local_is_sorted;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace sparsity
}  // namespace omp
}  // namespace kernels
}  // namespace gko
