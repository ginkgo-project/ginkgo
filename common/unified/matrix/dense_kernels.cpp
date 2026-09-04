// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace dense {


template <typename ValueType>
void compute_max_nnz_per_row(std::shared_ptr<const DefaultExecutor> exec,
                             matrix::view::dense<const ValueType> source,
                             size_type& result)
{
    array<size_type> partial{exec, source.size[0] + 1};
    count_nonzeros_per_row(exec, source, partial.get_data());
    run_kernel_reduction(
        exec, [] GKO_KERNEL(auto i, auto partial) { return partial[i]; },
        GKO_KERNEL_REDUCE_MAX(size_type), partial.get_data() + source.size[0],
        source.size[0], partial);
    result = get_element(partial, source.size[0]);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,
                        matrix::view::dense<const ValueType> source,
                        size_type slice_size, size_type stride_factor,
                        size_type* slice_sets, size_type* slice_lengths)
{
    const auto num_rows = source.size[0];
    array<size_type> row_nnz{exec, num_rows};
    count_nonzeros_per_row(exec, source, row_nnz.get_data());
    const auto num_slices =
        static_cast<size_type>(ceildiv(num_rows, slice_size));
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto slice, auto local_row, auto row_nnz, auto slice_size,
                      auto stride_factor, auto num_rows) {
            const auto row = slice * slice_size + local_row;
            return row < num_rows ? static_cast<size_type>(
                                        ceildiv(row_nnz[row], stride_factor) *
                                        stride_factor)
                                  : size_type{};
        },
        GKO_KERNEL_REDUCE_MAX(size_type), slice_lengths, 1,
        gko::dim<2>{num_slices, slice_size}, row_nnz, slice_size, stride_factor,
        num_rows);
    exec->copy(num_slices, slice_lengths, slice_sets);
    components::prefix_sum_nonnegative(exec, slice_sets, num_slices + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(std::shared_ptr<const DefaultExecutor> exec,
                            matrix::view::dense<const ValueType> mtx,
                            IndexType* result)
{
    run_kernel_row_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto j, auto mtx) {
            return is_nonzero(mtx(i, j)) ? 1 : 0;
        },
        GKO_KERNEL_REDUCE_SUM(IndexType), result, 1, mtx.size, mtx);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL_SIZE_T);


}  // namespace dense
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
