/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <ginkgo/core/distributed/partition.hpp>

#include "core/distributed/partition_kernels.hpp"


namespace gko {
namespace experimental {
namespace distributed {
namespace partition {


GKO_REGISTER_OPERATION(count_ranges, partition::count_ranges);
GKO_REGISTER_OPERATION(build_from_mapping, partition::build_from_mapping);
GKO_REGISTER_OPERATION(build_from_contiguous, partition::build_from_contiguous);
GKO_REGISTER_OPERATION(build_ranges_from_global_size,
                       partition::build_ranges_from_global_size);
GKO_REGISTER_OPERATION(build_starting_indices,
                       partition::build_starting_indices);
GKO_REGISTER_OPERATION(has_ordered_parts, partition::has_ordered_parts);


}  // namespace partition


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
Partition<LocalIndexType, GlobalIndexType>::build_from_mapping(
    std::shared_ptr<const Executor> exec, const array<comm_index_type>& mapping,
    comm_index_type num_parts)
{
    auto local_mapping = make_temporary_clone(exec, &mapping);
    size_type num_ranges{};
    exec->run(partition::make_count_ranges(*local_mapping.get(), num_ranges));
    auto result = Partition::create(exec, num_parts, num_ranges);
    exec->run(partition::make_build_from_mapping(*local_mapping.get(),
                                                 result->offsets_.get_data(),
                                                 result->part_ids_.get_data()));
    result->finalize_construction();
    return result;
}


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
Partition<LocalIndexType, GlobalIndexType>::build_from_contiguous(
    std::shared_ptr<const Executor> exec, const array<GlobalIndexType>& ranges)
{
    auto local_ranges = make_temporary_clone(exec, &ranges);
    auto result = Partition::create(
        exec, static_cast<comm_index_type>(ranges.get_num_elems() - 1),
        ranges.get_num_elems() - 1);
    exec->run(partition::make_build_from_contiguous(
        *local_ranges.get(), result->offsets_.get_data(),
        result->part_ids_.get_data()));
    result->finalize_construction();
    return result;
}


template <typename LocalIndexType, typename GlobalIndexType>
std::unique_ptr<Partition<LocalIndexType, GlobalIndexType>>
Partition<LocalIndexType, GlobalIndexType>::build_from_global_size_uniform(
    std::shared_ptr<const Executor> exec, comm_index_type num_parts,
    GlobalIndexType global_size)
{
    array<GlobalIndexType> ranges(exec, num_parts + 1);
    exec->run(partition::make_build_ranges_from_global_size(
        num_parts, global_size, ranges));
    return Partition::build_from_contiguous(exec, ranges);
}


template <typename LocalIndexType, typename GlobalIndexType>
void Partition<LocalIndexType, GlobalIndexType>::finalize_construction()
{
    auto exec = offsets_.get_executor();
    exec->run(partition::make_build_starting_indices(
        offsets_.get_const_data(), part_ids_.get_const_data(), get_num_ranges(),
        get_num_parts(), num_empty_parts_, starting_indices_.get_data(),
        part_sizes_.get_data()));
    size_ = offsets_.get_executor()->copy_val_to_host(
        offsets_.get_const_data() + get_num_ranges());
}


template <typename LocalIndexType, typename GlobalIndexType>
bool Partition<LocalIndexType, GlobalIndexType>::has_connected_parts()
{
    return this->get_num_parts() - this->get_num_empty_parts() ==
           this->get_num_ranges();
}


template <typename LocalIndexType, typename GlobalIndexType>
bool Partition<LocalIndexType, GlobalIndexType>::has_ordered_parts()
{
    if (this->has_connected_parts()) {
        auto exec = this->get_executor();
        bool has_ordered_parts;
        exec->run(partition::make_has_ordered_parts(this, &has_ordered_parts));
        return has_ordered_parts;
    } else {
        return false;
    }
}


#define GKO_DECLARE_PARTITION(_local, _global) class Partition<_local, _global>
GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(GKO_DECLARE_PARTITION);


}  // namespace distributed
}  // namespace experimental
}  // namespace gko
