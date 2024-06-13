// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/index_set.hpp>


#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/array_access.hpp"
#include "core/base/index_set_kernels.hpp"


namespace gko {
namespace idx_set {


GKO_REGISTER_OPERATION(to_global_indices, idx_set::to_global_indices);
GKO_REGISTER_OPERATION(populate_subsets, idx_set::populate_subsets);
GKO_REGISTER_OPERATION(global_to_local, idx_set::global_to_local);
GKO_REGISTER_OPERATION(local_to_global, idx_set::local_to_global);


}  // namespace idx_set


template <typename IndexType>
void index_set<IndexType>::populate_subsets(
    const gko::array<IndexType>& indices, const bool is_sorted)
{
    auto exec = this->get_executor();
    this->num_stored_indices_ = indices.get_size();
    exec->run(idx_set::make_populate_subsets(
        this->index_space_size_, &indices, &this->subsets_begin_,
        &this->subsets_end_, &this->superset_cumulative_indices_, is_sorted));
}


template <typename IndexType>
bool index_set<IndexType>::contains(const IndexType input_index) const
{
    auto local_index = this->get_local_index(input_index);
    return local_index != invalid_index<IndexType>();
}


template <typename IndexType>
IndexType index_set<IndexType>::get_global_index(const IndexType index) const
{
    auto exec = this->get_executor();
    const auto local_idx =
        array<IndexType>(exec, std::initializer_list<IndexType>{index});
    auto global_idx =
        array<IndexType>(exec, this->map_local_to_global(local_idx, true));

    return get_element(global_idx, 0);
}


template <typename IndexType>
IndexType index_set<IndexType>::get_local_index(const IndexType index) const
{
    auto exec = this->get_executor();
    const auto global_idx =
        array<IndexType>(exec, std::initializer_list<IndexType>{index});
    auto local_idx =
        array<IndexType>(exec, this->map_global_to_local(global_idx, true));

    return get_element(local_idx, 0);
}


template <typename IndexType>
array<IndexType> index_set<IndexType>::to_global_indices() const
{
    auto exec = this->get_executor();
    auto num_elems =
        get_element(this->superset_cumulative_indices_,
                    this->superset_cumulative_indices_.get_size() - 1);
    auto decomp_indices = gko::array<IndexType>(exec, num_elems);
    exec->run(idx_set::make_to_global_indices(
        this->get_num_subsets(), this->get_subsets_begin(),
        this->get_subsets_end(), this->get_superset_indices(),
        decomp_indices.get_data()));

    return decomp_indices;
}


template <typename IndexType>
array<IndexType> index_set<IndexType>::map_local_to_global(
    const array<IndexType>& local_indices, const bool is_sorted) const
{
    auto exec = this->get_executor();
    auto global_indices = gko::array<IndexType>(exec, local_indices.get_size());

    GKO_ASSERT(this->get_num_subsets() >= 1);
    exec->run(idx_set::make_local_to_global(
        this->get_num_subsets(), this->get_subsets_begin(),
        this->get_superset_indices(),
        static_cast<IndexType>(local_indices.get_size()),
        local_indices.get_const_data(), global_indices.get_data(), is_sorted));
    return global_indices;
}


template <typename IndexType>
array<IndexType> index_set<IndexType>::map_global_to_local(
    const array<IndexType>& global_indices, const bool is_sorted) const
{
    auto exec = this->get_executor();
    auto local_indices = gko::array<IndexType>(exec, global_indices.get_size());

    GKO_ASSERT(this->get_num_subsets() >= 1);
    exec->run(idx_set::make_global_to_local(
        this->index_space_size_, this->get_num_subsets(),
        this->get_subsets_begin(), this->get_subsets_end(),
        this->get_superset_indices(),
        static_cast<IndexType>(local_indices.get_size()),
        global_indices.get_const_data(), local_indices.get_data(), is_sorted));
    return local_indices;
}


#define GKO_DECLARE_INDEX_SET(_type) class index_set<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET);


}  // namespace gko
