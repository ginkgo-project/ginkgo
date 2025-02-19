// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest_kernels.hpp"

#include "common/unified/base/kernel_launch.hpp"
#include "core/base/index_range.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace elimination_forest {


template <typename IndexType>
void map_postorder(std::shared_ptr<const DefaultExecutor> exec,
                   const IndexType* parents, const IndexType* child_ptrs,
                   const IndexType* children, IndexType size,
                   const IndexType* inv_postorder, IndexType* postorder_parents,
                   IndexType* postorder_child_ptrs,
                   IndexType* postorder_children)
{
    // map parents and child counts (including pseudo-root, thus + 1)
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto parents, auto child_ptrs, auto size,
                      auto inv_postorder, auto postorder_parents,
                      auto postorder_child_ptrs) {
            // special case: pseudo-root doesn't have postorder entry or parent
            const auto postorder_i = i == size ? size : inv_postorder[i];
            if (i < size) {
                const auto parent = parents[i];
                postorder_parents[inv_postorder[i]] =
                    parent == size ? size : inv_postorder[parent];
            }
            postorder_child_ptrs[postorder_i] =
                child_ptrs[i + 1] - child_ptrs[i];
        },
        static_cast<size_type>(size + 1), parents, child_ptrs, size,
        inv_postorder, postorder_parents, postorder_child_ptrs);
    // build postorder_child_ptrs from sizes
    components::prefix_sum_nonnegative(exec, postorder_child_ptrs,
                                       static_cast<size_type>(size + 2));
    // now map children for all nodes (including pseudo-root, thus + 1)
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto child_ptrs, auto children, auto size,
                      auto inv_postorder, auto postorder_child_ptrs,
                      auto postorder_children) {
            const auto postorder_i = i == size ? size : inv_postorder[i];
            const auto in_begin = child_ptrs[i];
            const auto in_end = child_ptrs[i + 1];
            auto out_idx = postorder_child_ptrs[postorder_i];
            for (const auto child : irange{in_begin, in_end}) {
                postorder_children[out_idx] = inv_postorder[children[child]];
                out_idx++;
            }
        },
        static_cast<size_type>(size + 1), child_ptrs, children, size,
        inv_postorder, postorder_child_ptrs, postorder_children);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_MAP_POSTORDER);


template <typename IndexType>
void pointer_double(std::shared_ptr<const DefaultExecutor> exec,
                    const IndexType* input, IndexType size, IndexType* output)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto input, auto size, auto output) {
            const auto target = input[i];
            output[i] = target == size ? size : input[target];
        },
        static_cast<size_type>(size), input, size, output);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_POINTER_DOUBLE);


}  // namespace elimination_forest
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
