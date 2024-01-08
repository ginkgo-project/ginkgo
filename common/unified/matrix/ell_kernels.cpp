// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/ell_kernels.hpp"


#include <ginkgo/core/base/math.hpp>


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The Ell matrix format namespace.
 *
 * @ingroup ell
 */
namespace ell {


template <typename IndexType>
void compute_max_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                         const array<IndexType>& row_ptrs, size_type& max_nnz)
{
    array<size_type> result{exec, 1};
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto row_ptrs) {
            return row_ptrs[i + 1] - row_ptrs[i];
        },
        GKO_KERNEL_REDUCE_MAX(size_type), result.get_data(),
        row_ptrs.get_size() - 1, row_ptrs);
    max_nnz = get_element(result, 0);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         const int64* row_ptrs,
                         matrix::Ell<ValueType, IndexType>* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto row, auto in_cols, auto in_vals, auto row_ptrs,
                      auto stride, auto num_cols, auto cols, auto values) {
            const auto begin = row_ptrs[row];
            const auto end = row_ptrs[row + 1];
            auto out_idx = row;
            for (auto i = begin; i < begin + num_cols; i++) {
                cols[out_idx] =
                    i < end ? in_cols[i] : invalid_index<IndexType>();
                values[out_idx] = i < end ? in_vals[i] : zero(values[out_idx]);
                out_idx += stride;
            }
        },
        output->get_size()[0], data.get_const_col_idxs(),
        data.get_const_values(), row_ptrs, output->get_stride(),
        output->get_num_stored_elements_per_row(), output->get_col_idxs(),
        output->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Ell<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    // ELL is stored in column-major, so we swap row and column parameters
    run_kernel(
        exec,
        [] GKO_KERNEL(auto ell_col, auto row, auto ell_stride, auto in_cols,
                      auto in_vals, auto out) {
            const auto ell_idx = ell_col * ell_stride + row;
            const auto col = in_cols[ell_idx];
            const auto val = in_vals[ell_idx];
            if (col != invalid_index<IndexType>()) {
                out(row, col) = val;
            }
        },
        dim<2>{source->get_num_stored_elements_per_row(),
               source->get_size()[0]},
        static_cast<int64>(source->get_stride()), source->get_const_col_idxs(),
        source->get_const_values(), result);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::Ell<ValueType, IndexType>* source,
          matrix::Ell<ValueType, IndexType>* result)
{
    // ELL is stored in column-major, so we swap row and column parameters
    run_kernel(
        exec,
        [] GKO_KERNEL(auto ell_col, auto row, auto in_ell_stride, auto in_cols,
                      auto in_vals, auto out_ell_stride, auto out_cols,
                      auto out_vals) {
            const auto in = row + ell_col * in_ell_stride;
            const auto out = row + ell_col * out_ell_stride;
            out_cols[out] = in_cols[in];
            out_vals[out] = in_vals[in];
        },
        dim<2>{source->get_num_stored_elements_per_row(),
               source->get_size()[0]},
        static_cast<int64>(source->get_stride()), source->get_const_col_idxs(),
        source->get_const_values(), static_cast<int64>(result->get_stride()),
        result->get_col_idxs(), result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_COPY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Ell<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    // ELL is stored in column-major, so we swap row and column parameters
    run_kernel(
        exec,
        [] GKO_KERNEL(auto ell_col, auto row, auto ell_stride, auto in_cols,
                      auto in_vals, auto out_row_ptrs, auto out_cols,
                      auto out_vals) {
            const auto ell_idx = ell_col * ell_stride + row;
            const auto row_begin = out_row_ptrs[row];
            const auto row_size = out_row_ptrs[row + 1] - row_begin;
            if (ell_col < row_size) {
                out_cols[row_begin + ell_col] = in_cols[ell_idx];
                out_vals[row_begin + ell_col] = in_vals[ell_idx];
            }
        },
        dim<2>{source->get_num_stored_elements_per_row(),
               source->get_size()[0]},
        static_cast<int64>(source->get_stride()), source->get_const_col_idxs(),
        source->get_const_values(), result->get_row_ptrs(),
        result->get_col_idxs(), result->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Ell<ValueType, IndexType>* source,
                            IndexType* result)
{
    // ELL is stored in column-major, so we swap row and column parameters
    run_kernel_col_reduction(
        exec,
        [] GKO_KERNEL(auto ell_col, auto row, auto ell_stride, auto in_cols) {
            const auto ell_idx = ell_col * ell_stride + row;
            return in_cols[ell_idx] != invalid_index<IndexType>() ? 1 : 0;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), result,
        dim<2>{source->get_num_stored_elements_per_row(),
               source->get_size()[0]},
        static_cast<int64>(source->get_stride()), source->get_const_col_idxs());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_COUNT_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Ell<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    // ELL is stored in column-major, so we swap row and column parameters
    run_kernel(
        exec,
        [] GKO_KERNEL(auto ell_col, auto row, auto ell_stride, auto in_cols,
                      auto in_vals, auto out_vals) {
            const auto ell_idx = ell_col * ell_stride + row;
            const auto col = in_cols[ell_idx];
            const auto val = in_vals[ell_idx];
            if (row == col) {
                out_vals[row] = val;
            }
        },
        dim<2>{orig->get_num_stored_elements_per_row(), orig->get_size()[0]},
        static_cast<int64>(orig->get_stride()), orig->get_const_col_idxs(),
        orig->get_const_values(), diag->get_values());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL);


}  // namespace ell
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
