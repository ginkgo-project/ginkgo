// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"

#include <ginkgo/core/base/types.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"
#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace components {


template <typename IndexType, typename RowPtrType>
void convert_ptrs_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                          const RowPtrType* ptrs, size_type num_blocks,
                          IndexType* idxs)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto block, auto ptrs, auto idxs) {
            auto begin = ptrs[block];
            auto end = ptrs[block + 1];
            for (auto i = begin; i < end; i++) {
                idxs[i] = block;
            }
        },
        num_blocks, ptrs, idxs);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS64);


template <typename IndexType, typename RowPtrType>
void convert_idxs_to_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                          const IndexType* idxs, size_type num_idxs,
                          size_type num_blocks, RowPtrType* ptrs)
{
    if (num_idxs == 0) {
        fill_array(exec, ptrs, num_blocks + 1, RowPtrType{});
    } else {
        // one work item per block: ptrs[block] = number of idxs < block,
        // found by binary search in the sorted idxs. The work per item does
        // not depend on the gaps between consecutive idxs (empty blocks).
        run_kernel(
            exec,
            [] GKO_KERNEL(auto block, auto num_idxs, auto idxs, auto ptrs) {
                int64 lo = 0;
                int64 hi = num_idxs;
                while (lo < hi) {
                    const auto mid = lo + (hi - lo) / 2;
                    if (idxs[mid] < block) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                ptrs[block] = static_cast<RowPtrType>(lo);
            },
            num_blocks + 1, static_cast<int64>(num_idxs), idxs, ptrs);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_IDXS_TO_PTRS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_IDXS_TO_PTRS64);


template <typename RowPtrType>
void convert_ptrs_to_sizes(std::shared_ptr<const DefaultExecutor> exec,
                           const RowPtrType* ptrs, size_type num_blocks,
                           size_type* sizes)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto block, auto ptrs, auto sizes) {
            sizes[block] = ptrs[block + 1] - ptrs[block];
        },
        num_blocks, ptrs, sizes);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_SIZES);


template <typename RowPtrType>
void compute_max_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                         const array<RowPtrType>& row_ptrs, size_type& max_nnz)
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

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_COMPUTE_MAX_ROW_NNZ);


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
