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

#include <ginkgo/core/distributed/partition.hpp>


#include "core/distributed/partition_kernels.hpp"


namespace gko {
namespace distributed {
namespace partition {


GKO_REGISTER_OPERATION(count_ranges, partition::count_ranges);
GKO_REGISTER_OPERATION(build_from_mapping, partition::build_from_mapping);
GKO_REGISTER_OPERATION(build_from_contiguous, partition::build_from_contiguous);
GKO_REGISTER_OPERATION(build_ranks, partition::build_ranks);


}  // namespace partition


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_mapping(
    std::shared_ptr<const Executor> exec, const Array<comm_index_type> &mapping,
    comm_index_type num_parts)
{
    auto local_mapping = make_temporary_clone(exec, &mapping);
    size_type num_ranges{};
    exec->run(partition::make_count_ranges(*local_mapping.get(), num_ranges));
    auto result = Partition::create(exec, num_parts, num_ranges);
    exec->run(
        partition::make_build_from_mapping(*local_mapping.get(), result.get()));
    result->compute_range_ranks();
    return result;
}


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_contiguous(
    std::shared_ptr<const Executor> exec,
    const Array<global_index_type> &ranges)
{
    auto local_ranges = make_temporary_clone(exec, &ranges);
    auto result = Partition::create(
        exec, static_cast<comm_index_type>(ranges.get_num_elems() - 1),
        ranges.get_num_elems() - 1);
    exec->run(partition::make_build_from_contiguous(*local_ranges.get(),
                                                    result.get()));
    result->compute_range_ranks();
    return result;
}


template <typename LocalIndexType>
void Partition<LocalIndexType>::compute_range_ranks()
{
    auto exec = offsets_.get_executor();
    exec->run(partition::make_build_ranks(
        offsets_.get_const_data(), part_ids_.get_const_data(), get_num_ranges(),
        get_num_parts(), ranks_.get_data(), part_sizes_.get_data()));
}


template <typename LocalIndexType>
void Partition<LocalIndexType>::validate_data() const
{
    PolymorphicObject::validate_data();
    const auto exec = this->get_executor();
    // executors
    GKO_VALIDATION_CHECK(offsets_.get_executor() == exec);
    GKO_VALIDATION_CHECK(ranks_.get_executor() == exec);
    GKO_VALIDATION_CHECK(part_sizes_.get_executor() == exec);
    GKO_VALIDATION_CHECK(part_ids_.get_executor() == exec);
    // sizes
    const auto num_ranges = this->get_num_ranges();
    const auto num_parts = part_sizes_.get_num_elems();
    GKO_VALIDATION_CHECK(num_ranges >= 0);
    GKO_VALIDATION_CHECK(ranks_.get_num_elems() == num_ranges);
    GKO_VALIDATION_CHECK(part_ids_.get_num_elems() == num_ranges);
    GKO_VALIDATION_CHECK(part_sizes_.get_num_elems() == num_parts);
    // check range offsets: non-descending starting at 0
    Array<global_index_type> host_offsets(exec->get_master(), offsets_);
    const auto host_offset_ptr = host_offsets.get_const_data();
    GKO_VALIDATION_CHECK(host_offset_ptr[0] == 0);
    GKO_VALIDATION_CHECK_NAMED(
        "offsets need to be non-descending",
        std::is_sorted(host_offset_ptr, host_offset_ptr + (num_ranges + 1)));
    // check part IDs: in range [0, num_parts)
    Array<comm_index_type> host_part_ids(exec->get_master(), part_ids_);
    const auto host_part_id_ptr = host_part_ids.get_const_data();
    GKO_VALIDATION_CHECK_NAMED(
        "part IDs need to be in range",
        std::all_of(host_part_id_ptr, host_part_id_ptr + num_ranges,
                    [&](auto id) { return id >= 0 && id < num_parts; }));
    // check ranks and part sizes
    std::vector<global_index_type> partial_part_sizes(num_parts);
    Array<local_index_type> host_ranks(exec->get_master(), ranks_);
    Array<local_index_type> host_part_sizes(exec->get_master(), part_sizes_);
    const auto host_rank_ptr = host_ranks.get_const_data();
    const auto host_part_size_ptr = host_part_sizes.get_const_data();
    for (size_type i = 0; i < num_ranges; i++) {
        const auto part = host_part_id_ptr[i];
        const auto rank = host_rank_ptr[i];
        const auto range_size = host_offset_ptr[i + 1] - host_offset_ptr[i];
        GKO_VALIDATION_CHECK_NAMED("computed and stored range ranks must match",
                                   rank == partial_part_sizes[part]);
        partial_part_sizes[part] += range_size;
    }
    GKO_VALIDATION_CHECK_NAMED(
        "computed and stored part sizes must match",
        std::equal(partial_part_sizes.begin(), partial_part_sizes.end(),
                   host_part_size_ptr));
}


#define GKO_DECLARE_PARTITION(_type) class Partition<_type>
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PARTITION);


}  // namespace distributed
}  // namespace gko