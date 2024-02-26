// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Sellp matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


template <typename IndexType>
void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,
                        const array<IndexType>& row_ptrs, size_type slice_size,
                        size_type stride_factor, size_type* slice_sets,
                        size_type* slice_lengths)
{
    const auto num_rows = row_ptrs.get_size() - 1;
    const auto num_slices =
        static_cast<size_type>(ceildiv(num_rows, slice_size));
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto slice, auto local_row, auto row_ptrs,
                      auto slice_size, auto stride_factor, auto num_rows) {
            const auto row = slice * slice_size + local_row;
            return row < num_rows
                       ? static_cast<size_type>(
                             ceildiv(row_ptrs[row + 1] - row_ptrs[row],
                                     stride_factor) *
                             stride_factor)
                       : size_type{};
        },
        GKO_KERNEL_REDUCE_MAX(size_type), slice_lengths, 1,
        gko::dim<2>{num_slices, slice_size}, row_ptrs, slice_size,
        stride_factor, num_rows);
    exec->copy(num_slices, slice_lengths, slice_sets);
    components::prefix_sum_nonnegative(exec, slice_sets, num_slices + 1);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_SELLP_COMPUTE_SLICE_SETS_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         const int64* row_ptrs,
                         matrix::Sellp<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_cols, auto in_vals, auto row_ptrs,
                      auto slice_size, auto slice_sets, auto cols,
                      auto values) {
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto out_idx = slice_begin * slice_size + local_row;
            for (auto i = row_begin; i < row_begin + slice_length; i++) {
                cols[out_idx] =
                    i < row_end ? in_cols[i] : invalid_index<IndexType>();
                values[out_idx] =
                    i < row_end ? in_vals[i] : zero(values[out_idx]);
                out_idx += slice_size;
            }
        },
        output->get_size()[0], data.get_const_col_idxs(),
        data.get_const_values(), row_ptrs, output->get_slice_size(),
        output->get_const_slice_sets(), output->get_col_idxs(),
        output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Sellp<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto slice_size, auto slice_sets, auto cols,
                      auto values, auto result) {
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto in_idx = slice_begin * slice_size + local_row;
            for (int64 i = 0; i < slice_length; i++) {
                const auto col = cols[in_idx];
                if (col != invalid_index<IndexType>()) {
                    result(row, cols[in_idx]) = values[in_idx];
                }
                in_idx += slice_size;
            }
        },
        source->get_size()[0], source->get_slice_size(),
        source->get_const_slice_sets(), source->get_const_col_idxs(),
        source->get_const_values(), result);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Sellp<ValueType, IndexType>* source,
                            IndexType* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto slice_size, auto slice_sets, auto cols,
                      auto result) {
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto in_idx = slice_begin * slice_size + local_row;
            IndexType row_nnz{};
            for (int64 i = 0; i < slice_length; i++) {
                row_nnz += cols[in_idx] != invalid_index<IndexType>() ? 1 : 0;
                in_idx += slice_size;
            }
            result[row] = row_nnz;
        },
        source->get_size()[0], source->get_slice_size(),
        source->get_const_slice_sets(), source->get_const_col_idxs(), result);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_COUNT_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Sellp<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto slice_size, auto slice_sets, auto cols,
                      auto values, auto out_row_ptrs, auto out_cols,
                      auto out_vals) {
            const auto row_begin = out_row_ptrs[row];
            const auto row_end = out_row_ptrs[row + 1];
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto in_idx = slice_begin * slice_size + local_row;
            for (auto i = row_begin; i < row_end; i++) {
                out_cols[i] = cols[in_idx];
                out_vals[i] = values[in_idx];
                in_idx += slice_size;
            }
        },
        source->get_size()[0], source->get_slice_size(),
        source->get_const_slice_sets(), source->get_const_col_idxs(),
        source->get_const_values(), result->get_row_ptrs(),
        result->get_col_idxs(), result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Sellp<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto slice_size, auto slice_sets, auto cols,
                      auto values, auto diag) {
            const auto slice = row / slice_size;
            const auto local_row = row % slice_size;
            const auto slice_begin = slice_sets[slice];
            const auto slice_end = slice_sets[slice + 1];
            const auto slice_length = slice_end - slice_begin;
            auto in_idx = slice_begin * slice_size + local_row;
            for (int64 i = 0; i < slice_length; i++) {
                if (row == cols[in_idx]) {
                    diag[row] = values[in_idx];
                    break;
                }
                in_idx += slice_size;
            }
        },
        orig->get_size()[0], orig->get_slice_size(),
        orig->get_const_slice_sets(), orig->get_const_col_idxs(),
        orig->get_const_values(), diag->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_EXTRACT_DIAGONAL_KERNEL);


}  // namespace sellp
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
