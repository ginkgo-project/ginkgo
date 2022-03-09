/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/distributed/vector_kernels.hpp"


#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>


#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace distributed_vector {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const distributed::Partition<LocalIndexType, GlobalIndexType>* partition,
    comm_index_type local_part, matrix::Dense<ValueType>* local_mtx)
{
    const auto* range_bounds = partition->get_range_bounds();
    const auto* range_starting_indices =
        partition->get_range_starting_indices();
    const auto* part_ids = partition->get_part_ids();
    const auto num_ranges = partition->get_num_ranges();
    auto policy =
        oneapi::dpl::execution::make_device_policy(*exec->get_queue());

    Array<size_type> range_id{exec, input.get_num_elems()};
    oneapi::dpl::upper_bound(policy, range_bounds + 1,
                             range_bounds + num_ranges + 1,
                             input.get_const_row_idxs(),
                             input.get_const_row_idxs() + input.get_num_elems(),
                             range_id.get_data());

    // write values with local rows into the local matrix at the correct index
    // this needs the following iterators:
    // - local_row_it: (global_row, range_id) -> local row index
    // - flat_idx_it: (index) -> flat index (row[index] * stride + col[index])
    //                           in local matrix values array
    // the flat_idx_it is used by the copy_if as a permutation index for the
    // values
    auto map_to_local_row = [range_bounds, range_starting_indices](
                                const auto& idx_range_id) -> LocalIndexType {
        const auto [idx, rid] = idx_range_id;
        return static_cast<LocalIndexType>(idx - range_bounds[rid]) +
               range_starting_indices[rid];
    };
    auto local_row_it = oneapi::dpl::make_transform_iterator(
        oneapi::dpl::make_zip_iterator(input.get_const_row_idxs(),
                                       range_id.get_data()),
        map_to_local_row);

    auto flat_idx_it = oneapi::dpl::make_permutation_iterator(
        local_mtx->get_values(),
        [local_row_it, cols = input.get_const_col_idxs(),
         stride = local_mtx->get_stride()](const auto i) {
            return local_row_it[i] * stride + cols[i];
        });

    auto is_local_row = [range_id = range_id.get_data(), part_ids,
                         local_part](const auto i) {
        return part_ids[range_id[i]] == local_part;
    };
    oneapi::dpl::copy_if(policy, input.get_const_values(),
                         input.get_const_values() + input.get_num_elems(),
                         flat_idx_it, is_local_row);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_VECTOR_BUILD_LOCAL);


}  // namespace distributed_vector
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
