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


#include <numeric>


#include <ginkgo/core/base/mpi.hpp>


#include "core/distributed/partition_kernels.hpp"


namespace gko {
namespace distributed {
namespace partition {


GKO_REGISTER_OPERATION(count_ranges, partition::count_ranges);
GKO_REGISTER_OPERATION(build_from_mapping, partition::build_from_mapping);
GKO_REGISTER_OPERATION(build_from_contiguous, partition::build_from_contiguous);
GKO_REGISTER_OPERATION(build_ranks, partition::build_ranks);
GKO_REGISTER_OPERATION(is_ordered, partition::is_ordered);
GKO_REGISTER_OPERATION(build_block_gathered_permute,
                       partition::build_block_gathered_permute);


}  // namespace partition


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_mapping(
    std::shared_ptr<const Executor> exec, const Array<comm_index_type>& mapping,
    comm_index_type num_parts)
{
    auto local_mapping = make_temporary_clone(exec, &mapping);
    size_type num_ranges{};
    exec->run(partition::make_count_ranges(*local_mapping.get(), num_ranges));
    auto result = Partition::create(exec, num_parts, num_ranges);
    exec->run(
        partition::make_build_from_mapping(*local_mapping.get(), result.get()));
    result->compute_range_ranks();
    result->compute_block_gather_permutation();
    return result;
}


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_contiguous(
    std::shared_ptr<const Executor> exec,
    const Array<global_index_type>& ranges)
{
    auto local_ranges = make_temporary_clone(exec, &ranges);
    auto result = Partition::create(
        exec, static_cast<comm_index_type>(ranges.get_num_elems() - 1),
        ranges.get_num_elems() - 1);
    exec->run(partition::make_build_from_contiguous(*local_ranges.get(),
                                                    result.get()));
    result->compute_range_ranks();
    result->compute_block_gather_permutation();
    return result;
}


template <typename LocalIndexType>
std::unique_ptr<Partition<LocalIndexType>>
Partition<LocalIndexType>::build_from_local_range(
    std::shared_ptr<const Executor> exec, local_index_type local_start,
    local_index_type local_end, std::shared_ptr<const mpi::communicator> comm)
{
    global_index_type range[2] = {static_cast<global_index_type>(local_start),
                                  static_cast<global_index_type>(local_end)};

    // make all range_ends available on each rank
    Array<global_index_type> ranges_start_end(exec->get_master(),
                                              comm->size() * 2);
    ranges_start_end.fill(0);
    mpi::all_gather(range, 2, ranges_start_end.get_data(), 2, comm);

    // remove duplicates
    Array<global_index_type> ranges(exec->get_master(), comm->size() + 1);
    auto ranges_se_data = ranges_start_end.get_const_data();
    ranges.get_data()[0] = ranges_se_data[0];
    for (int i = 1; i < ranges_start_end.get_num_elems() - 1; i += 2) {
        GKO_ASSERT_EQ(ranges_se_data[i], ranges_se_data[i + 1]);
        ranges.get_data()[i / 2 + 1] = ranges_se_data[i];
    }
    ranges.get_data()[ranges.get_num_elems() - 1] =
        ranges_se_data[ranges_start_end.get_num_elems() - 1];

    // move data to correct executor
    ranges.set_executor(exec);

    return Partition::build_from_contiguous(exec, ranges);
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
void Partition<LocalIndexType>::compute_block_gather_permutation(
    const bool recompute)
{
    if (block_gather_permutation_.get_num_elems() == 0 || recompute) {
        block_gather_permutation_.resize_and_reset(this->get_size());
        block_gather_permutation_.fill(-1);
        auto exec = block_gather_permutation_.get_executor();
        exec->run(partition::make_build_block_gathered_permute(
            this, block_gather_permutation_));
    }
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


template <typename LocalIndexType>
bool is_connected(const Partition<LocalIndexType>* partition)
{
    return partition->get_num_parts() == partition->get_num_ranges();
}

#define GKO_DECLARE_IS_CONNECTED(_type) \
    bool is_connected(const Partition<_type>* partition)
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_IS_CONNECTED);


template <typename LocalIndexType>
bool is_ordered(const Partition<LocalIndexType>* partition)
{
    if (is_connected(partition)) {
        auto exec = partition->get_executor();
        bool is_ordered;
        exec->run(partition::make_is_ordered(partition, &is_ordered));
        return is_ordered;
    } else {
        return false;
    }
}

#define GKO_DECLARE_IS_ORDERED(_type) \
    bool is_ordered(const Partition<_type>* partition)
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_IS_ORDERED);


}  // namespace distributed
}  // namespace gko
