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

#include "core/distributed/partition_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition {


void count_ranges(std::shared_ptr<const DefaultExecutor> exec,
                  const Array<comm_index_type>& mapping, size_type& num_ranges)
{
    Array<size_type> result{exec, 1};
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(auto i, auto mapping) {
            auto cur_part = mapping[i];
            auto prev_part = i == 0 ? comm_index_type{-1} : mapping[i - 1];
            return cur_part != prev_part ? 1 : 0;
        },
        [] GKO_KERNEL(auto a, auto b) { return a + b; },
        [] GKO_KERNEL(auto a) { return a; }, size_type{}, result.get_data(),
        mapping.get_num_elems(), mapping);
    num_ranges = exec->copy_val_to_host(result.get_const_data());
}


template <typename LocalIndexType>
void build_from_contiguous(std::shared_ptr<const DefaultExecutor> exec,
                           const Array<global_index_type>& ranges,
                           distributed::Partition<LocalIndexType>* partition)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto ranges, auto bounds, auto ids) {
            if (i == 0) {
                bounds[0] = 0;
            }
            bounds[i + 1] = ranges[i + 1];
            ids[i] = i;
        },
        ranges.get_num_elems() - 1, ranges, partition->get_range_bounds(),
        partition->get_part_ids());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_FROM_CONTIGUOUS);


template <typename LocalIndexType>
void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec,
                        const Array<comm_index_type>& mapping,
                        distributed::Partition<LocalIndexType>* partition)
{
    Array<size_type> range_index_ranks{exec, mapping.get_num_elems() + 1};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto mapping, auto output) {
            const auto prev_part = i > 0 ? mapping[i - 1] : comm_index_type{-1};
            const auto cur_part = mapping[i];
            output[i] = cur_part != prev_part ? 1 : 0;
        },
        mapping.get_num_elems(), mapping, range_index_ranks);
    components::prefix_sum(exec, range_index_ranks.get_data(),
                           mapping.get_num_elems() + 1);
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto size, auto mapping, auto prefix_sum,
                      auto ranges, auto range_parts) {
            const auto prev_part = i > 0 ? mapping[i - 1] : comm_index_type{-1};
            const auto cur_part = i < size ? mapping[i] : comm_index_type{-1};
            if (cur_part != prev_part) {
                auto out_idx = prefix_sum[i];
                ranges[out_idx] = i;
                if (i < size) {
                    range_parts[out_idx] = cur_part;
                }
            }
        },
        mapping.get_num_elems() + 1, mapping.get_num_elems(), mapping,
        range_index_ranks, partition->get_range_bounds(),
        partition->get_part_ids());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PARTITION_BUILD_FROM_MAPPING);


}  // namespace partition
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
