// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"

#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace omp {
namespace components {


template <typename IndexType, typename RowPtrType>
void convert_ptrs_to_idxs(std::shared_ptr<const DefaultExecutor> exec,
                          const RowPtrType* ptrs, size_type num_blocks,
                          IndexType* idxs)
{
#pragma omp parallel for
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


}  // namespace components
}  // namespace omp
}  // namespace kernels
}  // namespace gko
