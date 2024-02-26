// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_INDEX_SET_KERNELS_HPP_
#define GKO_CORE_BASE_INDEX_SET_KERNELS_HPP_


#include <ginkgo/core/base/index_set.hpp>


#include <memory>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL(IndexType)       \
    void compute_validity(std::shared_ptr<const DefaultExecutor> exec, \
                          const array<IndexType>* local_indices,       \
                          array<bool>* validity_array)

#define GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL(IndexType)       \
    void to_global_indices(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType num_subsets, const IndexType* subset_begin,     \
        const IndexType* subset_end, const IndexType* superset_indices, \
        IndexType* decomp_indices)

#define GKO_DECLARE_INDEX_SET_POPULATE_KERNEL(IndexType)                   \
    void populate_subsets(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        const IndexType index_space_size, const array<IndexType>* indices, \
        array<IndexType>* subset_begin, array<IndexType>* subset_end,      \
        array<IndexType>* superset_indices, const bool is_sorted)

#define GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL(IndexType)         \
    void global_to_local(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType index_space_size, const IndexType num_subsets,  \
        const IndexType* subset_begin, const IndexType* subset_end,     \
        const IndexType* superset_indices, const IndexType num_indices, \
        const IndexType* global_indices, IndexType* local_indices,      \
        const bool is_sorted)

#define GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL(IndexType)         \
    void local_to_global(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const IndexType num_subsets, const IndexType* subset_begin,     \
        const IndexType* superset_indices, const IndexType num_indices, \
        const IndexType* local_indices, IndexType* global_indices,      \
        const bool is_sorted)


#define GKO_DECLARE_ALL_AS_TEMPLATES                           \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL(IndexType);  \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL(IndexType); \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_POPULATE_KERNEL(IndexType);          \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL(IndexType);   \
    template <typename IndexType>                              \
    GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(idx_set, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_BASE_INDEX_SET_KERNELS_HPP_
