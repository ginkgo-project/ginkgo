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

#include <ginkgo/core/base/index_set.hpp>


#include <algorithm>
#include <iostream>
#include <mutex>
#include <vector>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/index_set_kernels.hpp"


namespace gko {
namespace index_set {


GKO_REGISTER_OPERATION(populate_subsets, index_set::populate_subsets);
GKO_REGISTER_OPERATION(global_to_local, index_set::global_to_local);
GKO_REGISTER_OPERATION(local_to_global, index_set::local_to_global);


}  // namespace index_set


template <typename IndexType>
void IndexSet<IndexType>::populate_subsets(const gko::Array<IndexType> &indices,
                                           const bool is_sorted)
{
    auto exec = this->get_executor();
    this->num_stored_indices_ = indices.get_num_elems();
    exec->run(index_set::make_populate_subsets(
        this->index_space_size_, &indices, &this->subsets_begin_,
        &this->subsets_end_, &this->superset_cumulative_indices_, is_sorted));
}


template <typename IndexType>
bool IndexSet<IndexType>::is_element(const IndexType input_index) const
{
    auto local_index = this->get_local_index(input_index);
    return local_index != invalid_index<IndexType>();
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_global_index(const IndexType index) const
{
    auto exec = this->get_executor();
    const auto local_idx =
        Array<IndexType>(exec, std::initializer_list<IndexType>{index});
    auto global_idx = Array<IndexType>(
        exec->get_master(), this->get_global_indices(local_idx, true));

    return global_idx.get_data()[0];
}


template <typename IndexType>
IndexType IndexSet<IndexType>::get_local_index(const IndexType index) const
{
    auto exec = this->get_executor();
    const auto global_idx =
        Array<IndexType>(exec, std::initializer_list<IndexType>{index});
    auto local_idx = Array<IndexType>(
        exec->get_master(), this->get_local_indices(global_idx, true));

    return local_idx.get_data()[0];
}


template <typename IndexType>
Array<IndexType> IndexSet<IndexType>::get_global_indices(
    const Array<IndexType> &local_indices, const bool is_sorted) const
{
    auto exec = this->get_executor();
    auto global_indices =
        gko::Array<IndexType>(exec, local_indices.get_num_elems());

    GKO_ASSERT(this->get_num_subsets() >= 1);
    exec->run(index_set::make_local_to_global(
        this->index_space_size_, &this->subsets_begin_, &this->subsets_end_,
        &this->superset_cumulative_indices_, &local_indices, &global_indices,
        is_sorted));
    return std::move(global_indices);
}


template <typename IndexType>
Array<IndexType> IndexSet<IndexType>::get_local_indices(
    const Array<IndexType> &global_indices, const bool is_sorted) const
{
    auto exec = this->get_executor();
    auto local_indices =
        gko::Array<IndexType>(exec, global_indices.get_num_elems());

    GKO_ASSERT(this->get_num_subsets() >= 1);
    exec->run(index_set::make_global_to_local(
        this->index_space_size_, &this->subsets_begin_, &this->subsets_end_,
        &this->superset_cumulative_indices_, &global_indices, &local_indices,
        is_sorted));
    return std::move(local_indices);
}


#define GKO_DECLARE_INDEX_SET(_type) class IndexSet<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET);


}  // namespace gko
