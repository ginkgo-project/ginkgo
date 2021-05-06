/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <algorithm>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/allocator.hpp>
#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/index_set_kernels.hpp"


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
namespace index_set {


template <typename IndexType>
void populate_subsets(std::shared_ptr<const DefaultExecutor> exec,
                      const IndexType index_space_size,
                      const Array<IndexType> *indices,
                      Array<IndexType> *subset_begin,
                      Array<IndexType> *subset_end,
                      Array<IndexType> *superset_indices, const bool is_sorted)
{
    auto num_indices = indices->get_num_elems();
    auto tmp_indices = gko::Array<IndexType>(*indices);
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
        if (i < num_indices) {
            tmp_subset_begin.push_back(tmp_indices.get_data()[i]);
        }
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
    *subset_begin = std::move(gko::Array<IndexType>(
        exec, tmp_subset_begin.data(),
        tmp_subset_begin.data() + tmp_subset_begin.size()));
    *subset_end = std::move(
        gko::Array<IndexType>(exec, tmp_subset_end.data(),
                              tmp_subset_end.data() + tmp_subset_end.size()));
    *superset_indices = std::move(gko::Array<IndexType>(
        exec, tmp_subset_superset_index.data(),
        tmp_subset_superset_index.data() + tmp_subset_superset_index.size()));
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET_POPULATE_KERNEL);


template <typename IndexType>
void global_to_local(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType index_space_size,
                     const Array<IndexType> *subset_begin,
                     const Array<IndexType> *subset_end,
                     const Array<IndexType> *superset_indices,
                     const Array<IndexType> *global_indices,
                     Array<IndexType> *local_indices, const bool is_sorted)
{
    IndexType shifted_bucket = 0;
    // Loop over all the query indices.
    for (size_type i = 0; i < global_indices->get_num_elems(); ++i) {
        // If the query indices are sorted, then we dont need to search in the
        // entire set, but can search only in the successive complement set of
        // the previous search
        if (!is_sorted) {
            shifted_bucket = 0;
        }
        auto index = global_indices->get_const_data()[i];
        GKO_ASSERT(index < index_space_size);
        auto shifted_subset = &subset_begin->get_const_data()[shifted_bucket];
        auto bucket =
            std::distance(subset_begin->get_const_data(),
                          std::upper_bound(shifted_subset,
                                           subset_begin->get_const_data() +
                                               subset_begin->get_num_elems(),
                                           index));
        shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
        if (subset_end->get_const_data()[shifted_bucket] <= index) {
            local_indices->get_data()[i] = invalid_index<IndexType>();
        } else {
            local_indices->get_data()[i] =
                index - subset_begin->get_const_data()[shifted_bucket] +
                superset_indices->get_const_data()[shifted_bucket];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_GLOBAL_TO_LOCAL_KERNEL);


template <typename IndexType>
void local_to_global(std::shared_ptr<const DefaultExecutor> exec,
                     const IndexType index_space_size,
                     const Array<IndexType> *subset_begin,
                     const Array<IndexType> *subset_end,
                     const Array<IndexType> *superset_indices,
                     const Array<IndexType> *local_indices,
                     Array<IndexType> *global_indices, const bool is_sorted)
{
    IndexType shifted_bucket = 0;
    for (size_type i = 0; i < local_indices->get_num_elems(); ++i) {
        // If the query indices are sorted, then we dont need to search in the
        // entire set, but can search only in the successive complement set of
        // the previous search
        if (!is_sorted) {
            shifted_bucket = 0;
        }
        auto index = local_indices->get_const_data()[i];
        GKO_ASSERT(
            index <=
            (superset_indices
                 ->get_const_data()[superset_indices->get_num_elems() - 1]));
        auto shifted_superset =
            &superset_indices->get_const_data()[shifted_bucket];
        auto bucket = std::distance(
            superset_indices->get_const_data(),
            std::upper_bound(shifted_superset,
                             superset_indices->get_const_data() +
                                 superset_indices->get_num_elems(),
                             index));
        shifted_bucket = bucket == 0 ? 0 : (bucket - 1);
        global_indices->get_data()[i] =
            subset_begin->get_const_data()[shifted_bucket] + index -
            superset_indices->get_const_data()[shifted_bucket];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_INDEX_SET_LOCAL_TO_GLOBAL_KERNEL);


}  // namespace index_set
}  // namespace reference
}  // namespace kernels
}  // namespace gko
