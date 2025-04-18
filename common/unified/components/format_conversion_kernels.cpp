// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/format_conversion_kernels.hpp"

#include <ginkgo/core/base/types.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/components/bitvector.hpp"
#include "core/base/iterator_factory.hpp"
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
    const auto num_elements = exec->copy_val_to_host(ptrs + num_blocks);
    // transform the ptrs to a bitvector in unary delta encoding, i.e.
    // every row with n elements is encoded as 1 0 ... n times ... 0
    auto it = detail::make_transform_iterator(
        index_iterator<IndexType>{0},
        [ptrs] GKO_KERNEL(IndexType i) -> RowPtrType { return ptrs[i] + i; });
    auto bv = bitvector::from_sorted_indices(exec, it, num_blocks,
                                             num_blocks + num_elements);
    run_kernel(
        exec,
        [] GKO_KERNEL(RowPtrType i, auto bv, auto idxs) {
            if (!bv[i]) {
                auto rank = bv.get_rank(i);
                idxs[i - rank] = rank - 1;
            }
        },
        num_blocks + num_elements, bv.device_view(), idxs);
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
