// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"


#include <ginkgo/core/base/types.hpp>


#include "common/unified/base/kernel_launch.hpp"
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
        run_kernel(
            exec,
            [] GKO_KERNEL(auto i, auto num_idxs, auto num_blocks, auto idxs,
                          auto ptrs) {
                auto begin = i == 0 ? IndexType{} : idxs[i - 1];
                auto end = i == num_idxs ? num_blocks : idxs[i];
                for (auto block = begin; block < end; block++) {
                    ptrs[block + 1] = i;
                }
                if (i == 0) {
                    ptrs[0] = 0;
                }
            },
            num_idxs + 1, num_idxs, num_blocks, idxs, ptrs);
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


}  // namespace components
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
