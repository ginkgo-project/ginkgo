// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"


#include <ginkgo/core/base/types.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace components {


template <typename IndexType, typename RowPtrType>
void convert_ptrs_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                          const RowPtrType* ptrs, size_type num_blocks,
                          IndexType* idxs)
{
    for (size_type block = 0; block < num_blocks; block++) {
        auto begin = ptrs[block];
        auto end = ptrs[block + 1];
        for (auto i = begin; i < end; i++) {
            idxs[i] = block;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_IDXS64);


template <typename IndexType, typename RowPtrType>
void convert_idxs_to_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                          const IndexType* idxs, size_type num_idxs,
                          size_type num_blocks, RowPtrType* ptrs)
{
    fill_array(exec, ptrs, num_blocks + 1, RowPtrType{});
    for (size_type i = 0; i < num_idxs; i++) {
        ptrs[idxs[i]]++;
    }
    prefix_sum_nonnegative(exec, ptrs, num_blocks + 1);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_IDXS_TO_PTRS32);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_IDXS_TO_PTRS64);


template <typename RowPtrType>
void convert_ptrs_to_sizes(std::shared_ptr<const DefaultExecutor> exec,
                           const RowPtrType* ptrs, size_type num_blocks,
                           size_type* sizes)
{
    for (size_type block = 0; block < num_blocks; block++) {
        sizes[block] = ptrs[block + 1] - ptrs[block];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_CONVERT_PTRS_TO_SIZES);


}  // namespace components
}  // namespace reference
}  // namespace kernels
}  // namespace gko
