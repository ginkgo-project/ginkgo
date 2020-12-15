/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


}  // namespace index_set


template <typename IndexType>
void IndexSet<IndexType>::populate_subsets(const gko::Array<IndexType> &indices)
{
    auto exec = this->get_executor();

    auto num_indices = static_cast<IndexType>(indices.get_num_elems());

    exec->run(index_set::make_populate_subsets(
        this->index_space_size_, this->num_stored_indices_,
        indices.get_const_data(), num_indices, this->subsets_begin_.get_data(),
        this->subsets_end_.get_data(),
        this->superset_cumulative_indices_.get_data()));
}


template <typename IndexType>
bool IndexSet<IndexType>::is_element(const IndexType index) const
{}


template <typename IndexType>
void IndexSet<IndexType>::get_global_index(const IndexType &local_index,
                                           IndexType &global_index) const
{}


template <typename IndexType>
void IndexSet<IndexType>::get_local_index(const IndexType &global_index,
                                          IndexType &local_index) const
{}


#define GKO_DECLARE_INDEX_SET(_type) class IndexSet<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_INDEX_SET);


}  // namespace gko
