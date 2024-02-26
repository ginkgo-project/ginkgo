// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/index_set_kernels.hpp"


#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/allocator.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The Reference namespace.
 *
 * @ingroup reference
 */
namespace reference {
/**
 * @brief The index_set namespace.
 *
 * @ingroup index_set
 */
namespace idx_set {


template <typename IndexType>
void compute_validity(std::shared_ptr<const DefaultExecutor> exec,
                      const array<IndexType>* local_indices,
                      array<bool>* validity_array)
{
    auto num_elems = local_indices->get_size();
    for (size_type i = 0; i < num_elems; ++i) {
        validity_array->get_data()[i] =
            local_indices->get_const_data()[i] != invalid_index<IndexType>();
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_COMPUTE_VALIDITY_KERNEL);


template <typename IndexType>
void to_global_indices(std::shared_ptr<const DefaultExecutor> exec,
                       const IndexType num_subsets,
                       const IndexType* subset_begin,
                       const IndexType* subset_end,
                       const IndexType* superset_indices,
                       IndexType* decomp_indices)
{
    for (size_type subset = 0; subset < num_subsets; ++subset) {
        for (size_type i = 0;
             i < superset_indices[subset + 1] - superset_indices[subset]; ++i) {
            decomp_indices[superset_indices[subset] + i] =
                subset_begin[subset] + i;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_TO_GLOBAL_INDICES_KERNEL);


template <typename IndexType>
void populate_subsets(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType index_space_size,
                      const array<IndexType>* indices,
                      array<IndexType>* subset_begin,
                      array<IndexType>* subset_end,
                      array<IndexType>* superset_indices, const bool is_sorted)
{
    auto num_indices = indices->get_size();
    auto tmp_indices = gko::array<IndexType>(*indices);
    // Sort the indices if not sorted.
    if (!is_sorted) {
        std::sort(tmp_indices.get_data(), tmp_indices.get_data() + num_indices);
    }
    GKO_ASSERT(tmp_indices.get_const_data()[num_indices - 1] <=
               index_space_size);

    auto tmp_subset_begin = gko::vector<IndexType>(exec);
    auto tmp_subset_end = gko::vector<IndexType>(exec);
    auto tmp_subset_superset_index = gko::vector<IndexType>(exec);
    tmp_subset_begin.push_back(tmp_indices.get_data()[0]);
    tmp_subset_superset_index.push_back(0);
    // Detect subsets.
    for (size_type i = 1; i < num_indices; ++i) {
        if ((tmp_indices.get_data()[i] ==
             (tmp_indices.get_data()[i - 1] + 1)) ||
            (tmp_indices.get_data()[i] == tmp_indices.get_data()[i - 1])) {
            continue;
        }
        tmp_subset_end.push_back(tmp_indices.get_data()[i - 1] + 1);
        tmp_subset_superset_index.push_back(tmp_subset_superset_index.back() +
                                            tmp_subset_end.back() -
                                            tmp_subset_begin.back());
        tmp_subset_begin.push_back(tmp_indices.get_data()[i]);
    }
    tmp_subset_end.push_back(tmp_indices.get_data()[num_indices - 1] + 1);
    tmp_subset_superset_index.push_back(tmp_subset_superset_index.back() +
                                        tmp_subset_end.back() -
                                        tmp_subset_begin.back());

    // Make sure the sizes of the indices match and move them to their final
    // arrays.
    GKO_ASSERT(tmp_subset_begin.size() == tmp_subset_end.size());
    GKO_ASSERT((tmp_subset_begin.size() + 1) ==
               tmp_subset_superset_index.size());
    *subset_begin = std::move(gko::array<IndexType>(
        exec, tmp_subset_begin.data(),
        tmp_subset_begin.data() + tmp_subset_begin.size()));
    *subset_end = std::move(
        gko::array<IndexType>(exec, tmp_subset_end.data(),
                              tmp_subset_end.data() + tmp_subset_end.size()));
    *superset_indices = std::move(gko::array<IndexType>(
        exec, tmp_subset_superset_index.data(),
        tmp_subset_superset_index.data() + tmp_subset_superset_index.size()));
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET_POPULATE_KERNEL);


template <typename IndexType>
void global_to_local(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType index_space_size,
                     const IndexType num_subsets, const IndexType* subset_begin,
                     const IndexType* subset_end,
                     const IndexType* superset_indices,
                     const IndexType num_indices,
                     const IndexType* global_indices, IndexType* local_indices,
                     const bool is_sorted)
{
    IndexType shifted_bucket = 0;
    // Loop over all the query indices.
    for (size_type i = 0; i < num_indices; ++i) {
        // If the query indices are sorted, then we dont need to search in the
        // entire set, but can search only in the successive complement set of
        // the previous search
        if (!is_sorted) {
            shifted_bucket = 0;
        }
        auto index = global_indices[i];
        if (index < 0 || index >= index_space_size) {
            local_indices[i] = invalid_index<IndexType>();
            continue;
        }
        const auto shifted_subset = &subset_begin[shifted_bucket];
        auto bucket = std::distance(
            subset_begin, std::upper_bound(shifted_subset,
                                           subset_begin + num_subsets, index));
        shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
        if (subset_end[shifted_bucket] <= index ||
            index < subset_begin[shifted_bucket]) {
            local_indices[i] = invalid_index<IndexType>();
        } else {
            local_indices[i] = index - subset_begin[shifted_bucket] +
                               superset_indices[shifted_bucket];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL);


template <typename IndexType>
void local_to_global(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType num_subsets, const IndexType* subset_begin,
                     const IndexType* superset_indices,
                     const IndexType num_indices,
                     const IndexType* local_indices, IndexType* global_indices,
                     const bool is_sorted)
{
    IndexType shifted_bucket = 0;
    for (size_type i = 0; i < num_indices; ++i) {
        // If the query indices are sorted, then we dont need to search in the
        // entire set, but can search only in the successive complement set of
        // the previous search
        if (!is_sorted) {
            shifted_bucket = 0;
        }
        auto index = local_indices[i];
        if (index < 0 || index >= superset_indices[num_subsets]) {
            global_indices[i] = invalid_index<IndexType>();
            continue;
        }
        const auto shifted_superset = &superset_indices[shifted_bucket];
        auto bucket = std::distance(
            superset_indices,
            std::upper_bound(shifted_superset,
                             superset_indices + num_subsets + 1, index));
        shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
        global_indices[i] = subset_begin[shifted_bucket] + index -
                            superset_indices[shifted_bucket];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL);


}  // namespace idx_set
}  // namespace reference
}  // namespace kernels
}  // namespace gko
