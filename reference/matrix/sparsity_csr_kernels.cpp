// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/mixed_precision_types.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref SparsityCsr
 * @ingroup sparsity
 */
namespace sparsity_csr {


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::SparsityCsr<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    const auto val = static_cast<arithmetic_type>(a->get_const_value()[0]);

    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            auto temp_val = gko::zero<arithmetic_type>();
            for (size_type k = row_ptrs[row];
                 k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
                temp_val +=
                    val * static_cast<arithmetic_type>(b->at(col_idxs[k], j));
            }
            c->at(row, j) = static_cast<OutputValueType>(temp_val);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL);


template <typename MatrixValueType, typename InputValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::SparsityCsr<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;

    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    const auto valpha = static_cast<arithmetic_type>(alpha->at(0, 0));
    const auto vbeta = static_cast<arithmetic_type>(beta->at(0, 0));
    const auto val = static_cast<arithmetic_type>(a->get_const_value()[0]);

    for (size_type row = 0; row < a->get_size()[0]; ++row) {
        for (size_type j = 0; j < c->get_size()[1]; ++j) {
            auto temp_val = gko::zero<arithmetic_type>();
            for (size_type k = row_ptrs[row];
                 k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
                temp_val +=
                    val * static_cast<arithmetic_type>(b->at(col_idxs[k], j));
            }
            c->at(row, j) = static_cast<OutputValueType>(
                vbeta * static_cast<arithmetic_type>(c->at(row, j)) +
                valpha * temp_val);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::SparsityCsr<ValueType, IndexType>* input,
                   matrix::Dense<ValueType>* output)
{
    auto row_ptrs = input->get_const_row_ptrs();
    auto col_idxs = input->get_const_col_idxs();
    auto val = input->get_const_value()[0];

    for (size_type row = 0; row < input->get_size()[0]; ++row) {
        for (auto k = row_ptrs[row]; k < row_ptrs[row + 1]; ++k) {
            auto col = col_idxs[k];
            output->at(row, col) = val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void diagonal_element_prefix_sum(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* matrix,
    IndexType* prefix_sum)
{
    auto num_rows = matrix->get_size()[0];
    auto row_ptrs = matrix->get_const_row_ptrs();
    auto col_idxs = matrix->get_const_col_idxs();
    IndexType num_diag = 0;
    for (size_type i = 0; i < num_rows; ++i) {
        prefix_sum[i] = num_diag;
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            if (col_idxs[j] == i) {
                num_diag++;
            }
        }
    }
    prefix_sum[num_rows] = num_diag;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_DIAGONAL_ELEMENT_PREFIX_SUM_KERNEL);


template <typename ValueType, typename IndexType>
void remove_diagonal_elements(std::shared_ptr<const ReferenceExecutor> exec,
                              const IndexType* row_ptrs,
                              const IndexType* col_idxs,
                              const IndexType* diag_prefix_sum,
                              matrix::SparsityCsr<ValueType, IndexType>* matrix)
{
    auto num_rows = matrix->get_size()[0];
    auto adj_ptrs = matrix->get_row_ptrs();
    auto adj_idxs = matrix->get_col_idxs();
    size_type num_diag = 0;
    adj_ptrs[0] = row_ptrs[0];
    for (size_type i = 0; i < num_rows; ++i) {
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            if (col_idxs[j] == i) {
                num_diag++;
            }
        }
        adj_ptrs[i + 1] = row_ptrs[i + 1] - num_diag;
    }
    auto nnz = 0;
    for (size_type i = 0; i < num_rows; ++i) {
        for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
            if (col_idxs[j] != i) {
                adj_idxs[nnz] = col_idxs[j];
                nnz++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL);


template <typename IndexType>
inline void convert_sparsity_to_csc(size_type num_rows,
                                    const IndexType* row_ptrs,
                                    const IndexType* col_idxs,
                                    IndexType* row_idxs, IndexType* col_ptrs)
{
    for (size_type row = 0; row < num_rows; ++row) {
        for (auto i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = row;
        }
    }
}


template <typename ValueType, typename IndexType>
void transpose_and_transform(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* orig,
    matrix::SparsityCsr<ValueType, IndexType>* trans)
{
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();

    auto orig_num_cols = orig->get_size()[1];
    auto orig_num_rows = orig->get_size()[0];
    auto orig_nnz = orig_row_ptrs[orig_num_rows];

    components::fill_array(exec, trans_row_ptrs, orig_num_cols + 1,
                           IndexType{});
    for (size_type i = 0; i < orig_nnz; i++) {
        trans_row_ptrs[orig_col_idxs[i] + 1]++;
    }
    components::prefix_sum_nonnegative(exec, trans_row_ptrs + 1, orig_num_cols);

    convert_sparsity_to_csc(orig_num_rows, orig_row_ptrs, orig_col_idxs,
                            trans_col_idxs, trans_row_ptrs + 1);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::SparsityCsr<ValueType, IndexType>* orig,
               matrix::SparsityCsr<ValueType, IndexType>* trans)
{
    transpose_and_transform(exec, orig, trans);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::SparsityCsr<ValueType, IndexType>* to_sort)
{
    auto row_ptrs = to_sort->get_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    const auto number_rows = to_sort->get_size()[0];
    for (size_type i = 0; i < number_rows; ++i) {
        auto start_row_idx = row_ptrs[i];
        auto row_nnz = row_ptrs[i + 1] - start_row_idx;
        std::sort(col_idxs + start_row_idx, col_idxs + start_row_idx + row_nnz);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* to_check, bool* is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    for (size_type i = 0; i < size[0]; ++i) {
        for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
            if (col_idxs[idx - 1] > col_idxs[idx]) {
                *is_sorted = false;
                return;
            }
        }
    }
    *is_sorted = true;
    return;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace sparsity_csr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
