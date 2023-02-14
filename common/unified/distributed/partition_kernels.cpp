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

#include "core/distributed/partition_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition {


using experimental::distributed::comm_index_type;

void count_ranges(std::shared_ptr<const DefaultExecutor> exec,
                  const array<comm_index_type>& mapping, size_type& num_ranges)
{
    array<size_type> result{exec, 1};
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto mapping) {
            auto cur_part = mapping[i];
            auto prev_part = i == 0 ? comm_index_type{-1} : mapping[i - 1];
            return cur_part != prev_part ? 1 : 0;
        },
        GKO_KERNEL_REDUCE_SUM(size_type), result.get_data(),
        mapping.get_num_elems(), mapping);
    num_ranges = exec->copy_val_to_host(result.get_const_data());
}


template <typename GlobalIndexType>
void build_from_contiguous(std::shared_ptr<const DefaultExecutor> exec,
                           const array<GlobalIndexType>& ranges,
                           const array<comm_index_type>& part_id_mapping,
                           GlobalIndexType* range_bounds,
                           comm_index_type* part_ids)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto ranges, auto mapping, auto bounds, auto ids,
                      bool uses_mapping) {
            if (i == 0) {
                bounds[0] = 0;
            }
            bounds[i + 1] = ranges[i + 1];
            ids[i] = uses_mapping ? mapping[i] : i;
        },
        ranges.get_num_elems() - 1, ranges, part_id_mapping, range_bounds,
        part_ids, part_id_mapping.get_num_elems() > 0);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_CONTIGUOUS);


template <typename GlobalIndexType>
void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec,
                        const array<comm_index_type>& mapping,
                        GlobalIndexType* range_bounds,
                        comm_index_type* part_ids)
{
    array<size_type> range_starting_index{exec, mapping.get_num_elems() + 1};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto mapping, auto range_starting_index) {
            const auto prev_part =
                i > 0 ? mapping[i - 1] : invalid_index<comm_index_type>();
            const auto cur_part = mapping[i];
            range_starting_index[i] = cur_part != prev_part ? 1 : 0;
        },
        mapping.get_num_elems(), mapping, range_starting_index);
    components::prefix_sum(exec, range_starting_index.get_data(),
                           mapping.get_num_elems() + 1);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto size, auto mapping,
                      auto range_starting_index, auto ranges,
                      auto range_parts) {
            const auto prev_part =
                i > 0 ? mapping[i - 1] : invalid_index<comm_index_type>();
            const auto cur_part =
                i < size ? mapping[i] : invalid_index<comm_index_type>();
            if (cur_part != prev_part) {
                auto out_idx = range_starting_index[i];
                ranges[out_idx] = i;
                if (i < size) {
                    range_parts[out_idx] = cur_part;
                }
            }
        },
        mapping.get_num_elems() + 1, mapping.get_num_elems(), mapping,
        range_starting_index, range_bounds, part_ids);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_MAPPING);


template <typename GlobalIndexType>
void build_ranges_from_global_size(std::shared_ptr<const DefaultExecutor> exec,
                                   comm_index_type num_parts,
                                   GlobalIndexType global_size,
                                   array<GlobalIndexType>& ranges)
{
    const auto size_per_part = global_size / num_parts;
    const auto rest = global_size - (num_parts * size_per_part);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto size_per_part, auto rest, auto ranges) {
            ranges[i] = size_per_part + (i < rest ? 1 : 0);
        },
        ranges.get_num_elems() - 1, size_per_part, rest, ranges.get_data());
    components::prefix_sum(exec, ranges.get_data(), ranges.get_num_elems());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE);


template <typename LocalIndexType, typename GlobalIndexType>
void has_ordered_parts(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    bool* result)
{
    const auto part_ids = partition->get_part_ids();
    const auto num_ranges = partition->get_num_ranges();
    // it is necessary to use uint32 as a temporary result, since
    // bool can't be used with suffles
    array<uint32> result_uint32{exec, 1};
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, const auto part_ids) {
            return static_cast<uint32>(part_ids[i] < part_ids[i + 1]);
        },
        [] GKO_KERNEL(const auto a, const auto b) {
            return static_cast<uint32>(a && b);
        },
        [] GKO_KERNEL(const auto a) { return a; }, uint32(1),
        result_uint32.get_data(), num_ranges - 1, part_ids);
    *result = static_cast<bool>(
        exec->copy_val_to_host(result_uint32.get_const_data()));
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PARTITION_IS_ORDERED);


}  // namespace partition
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
