// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The SparsityCsr matrix format namespace.
 *
 * @ingroup sparsity_csr
 */
namespace sparsity_csr {


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::SparsityCsr<ValueType, IndexType>* input,
                   matrix::Dense<ValueType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto value,
                      auto output) {
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            const auto val = *value;
            for (auto nz = begin; nz < end; nz++) {
                output(row, col_idxs[nz]) = val;
            }
        },
        input->get_size()[0], input->get_const_row_ptrs(),
        input->get_const_col_idxs(), input->get_const_value(), output);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void diagonal_element_prefix_sum(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* matrix,
    IndexType* prefix_sum)
{
    const auto num_rows = matrix->get_size()[0];
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto row_ptrs, auto col_idxs, auto prefix_sum) {
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            IndexType count = 0;
            for (auto nz = begin; nz < end; nz++) {
                if (col_idxs[nz] == row) {
                    count++;
                }
            }
            prefix_sum[row] = count;
        },
        num_rows, matrix->get_const_row_ptrs(), matrix->get_const_col_idxs(),
        prefix_sum);
    components::prefix_sum_nonnegative(exec, prefix_sum, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_DIAGONAL_ELEMENT_PREFIX_SUM_KERNEL);


template <typename ValueType, typename IndexType>
void remove_diagonal_elements(std::shared_ptr<const DefaultExecutor> exec,
                              const IndexType* row_ptrs,
                              const IndexType* col_idxs,
                              const IndexType* diag_prefix_sum,
                              matrix::SparsityCsr<ValueType, IndexType>* matrix)
{
    const auto num_rows = matrix->get_size()[0];
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_row_ptrs, auto in_col_idxs,
                      auto diag_prefix_sum, auto out_row_ptrs,
                      auto out_col_idxs) {
            const auto in_begin = in_row_ptrs[row];
            const auto in_end = in_row_ptrs[row + 1];
            auto out_idx = in_begin - diag_prefix_sum[row];
            for (auto nz = in_begin; nz < in_end; nz++) {
                const auto col = in_col_idxs[nz];
                if (col != row) {
                    out_col_idxs[out_idx] = col;
                    out_idx++;
                }
            }
            if (row == 0) {
                out_row_ptrs[0] = 0;
            }
            out_row_ptrs[row + 1] = out_idx;
        },
        num_rows, row_ptrs, col_idxs, diag_prefix_sum, matrix->get_row_ptrs(),
        matrix->get_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL);


}  // namespace sparsity_csr
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
