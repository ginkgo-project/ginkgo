// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/elimination_forest.hpp"

#include <algorithm>
#include <memory>

#include <sycl/sycl.hpp>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/factorization/elimination_forest_kernels.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace elimination_forest {


template <typename IndexType>
void compute_skeleton_tree(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* row_ptrs, const IndexType* cols,
                           size_type size, IndexType* out_row_ptrs,
                           IndexType* out_cols) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SKELETON_TREE);


template <typename IndexType>
void compute(std::shared_ptr<const DefaultExecutor> exec,
             const IndexType* row_ptrs, const IndexType* cols, size_type size,
             gko::factorization::elimination_forest<IndexType>& forest)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELIMINATION_FOREST_COMPUTE);


template <typename ValueType, typename IndexType>
void from_factor(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::Csr<ValueType, IndexType>* factors,
                 IndexType* parents) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_FROM_FACTOR);


template <typename IndexType>
void compute_subtree_sizes(std::shared_ptr<const DefaultExecutor> exec,
                           const IndexType* child_ptrs,
                           const IndexType* children, IndexType size,
                           IndexType* subtree_sizes) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_SIZES);


template <typename IndexType>
void compute_subtree_euler_walk_sizes(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* child_ptrs,
    const IndexType* children, IndexType size,
    IndexType* subtree_sizes) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_SUBTREE_EULER_PATH_SIZES);


template <typename IndexType>
void compute_levels(std::shared_ptr<const DefaultExecutor> exec,
                    const IndexType* parents, IndexType size,
                    IndexType* levels) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_LEVELS);


template <typename IndexType>
void compute_postorder(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType* child_ptrs, const IndexType* children,
                       IndexType size, const IndexType* subtree_size,
                       IndexType* postorder,
                       IndexType* inv_postorder) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_POSTORDER);


template <typename IndexType>
void map_postorder(std::shared_ptr<const DefaultExecutor> exec,
                   const IndexType* parents, const IndexType* child_ptrs,
                   const IndexType* children, IndexType size,
                   const IndexType* postorder, const IndexType* inv_postorder,
                   IndexType* postorder_parents,
                   IndexType* postorder_child_ptrs,
                   IndexType* postorder_children) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_MAP_POSTORDER);


template <typename IndexType>
void compute_euler_walk(std::shared_ptr<const DefaultExecutor> exec,
                        const IndexType* child_ptrs, const IndexType* children,
                        IndexType size,
                        const IndexType* subtree_euler_tree_size,
                        const IndexType* levels, IndexType* euler_walk,
                        IndexType* first_visit,
                        IndexType* euler_levels) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_ELIMINATION_FOREST_COMPUTE_EULER_PATH);


}  // namespace elimination_forest
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
