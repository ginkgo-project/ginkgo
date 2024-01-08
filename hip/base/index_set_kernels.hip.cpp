// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/index_set_kernels.hpp"


#include <memory>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
/**
 * @brief The Hip namespace.
 *
 * @ingroup hip
 */
namespace hip {
/**
 * @brief The index_set namespace.
 *
 * @ingroup index_set
 */
namespace idx_set {


template <typename IndexType>
void to_global_indices(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType num_subsets,
                       const IndexType* subset_begin,
                       const IndexType* subset_end,
                       const IndexType* superset_indices,
                       IndexType* decomp_indices) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL);


template <typename IndexType>
void populate_subsets(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType index_space_size,
                      const array<IndexType>* indices,
                      array<IndexType>* subset_begin,
                      array<IndexType>* subset_end,
                      array<IndexType>* superset_indices,
                      const bool is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET_POPULATE_KERNEL);


template <typename IndexType>
void global_to_local(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType index_space_size,
                     const IndexType num_subsets, const IndexType* subset_begin,
                     const IndexType* subset_end,
                     const IndexType* superset_indices,
                     const IndexType num_indices,
                     const IndexType* global_indices, IndexType* local_indices,
                     const bool is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL);


template <typename IndexType>
void local_to_global(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType num_subsets, const IndexType* subset_begin,
                     const IndexType* superset_indices,
                     const IndexType num_indices,
                     const IndexType* local_indices, IndexType* global_indices,
                     const bool is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL);


}  // namespace idx_set
}  // namespace hip
}  // namespace kernels
}  // namespace gko
