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


namespace gko {
namespace kernels {
namespace dpcpp {
namespace partition {


template <typename LocalIndexType>
void build_starting_indices(std::shared_ptr<const DefaultExecutor> exec,
                            const global_index_type* range_offsets,
                            const int* range_parts, size_type num_ranges,
                            int num_parts, int& num_empty_parts,
                            LocalIndexType* ranks, LocalIndexType* sizes)
{
    Array<LocalIndexType> range_sizes{exec, num_ranges};
    // num_parts sentinel at the end
    Array<comm_index_type> tmp_part_ids{exec, num_ranges + 1};
    Array<global_index_type> permutation{exec, num_ranges};
    // set sizes to 0 in case of empty parts
    components::fill_array(exec, sizes, num_parts, LocalIndexType{});

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto num_ranges, auto num_parts,
                      auto range_offsets, auto range_parts, auto range_sizes,
                      auto tmp_part_ids, auto permutation) {
            if (i == 0) {
                // set sentinel value at the end
                tmp_part_ids[num_ranges] = num_parts;
            }
            range_sizes[i] = range_offsets[i + 1] - range_offsets[i];
            tmp_part_ids[i] = range_parts[i];
            permutation[i] = static_cast<global_index_type>(i);
        },
        num_ranges, num_ranges, num_parts, range_offsets, range_parts,
        range_sizes, tmp_part_ids, permutation);

    // group sizes by part ID
    // TODO oneDPL has stable_sort and views::zip
    // compute inclusive prefix sum for each part
    // TODO compute "row_ptrs" for tmp_part_ids
    // TODO compute prefix_sum over range_sizes
    // TODO compute adjacent differences, set -part_size at part boundaries
    // TODO compute prefix_sum again
    // write back the results
    // TODO this needs to be adapted to the output of the algorithm above
    // TODO count number of zeros in size and store in num_empty_parts
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto grouped_range_ranks, auto grouped_part_ids,
                      auto orig_idxs, auto ranks, auto sizes) {
            auto prev_part =
                i > 0 ? grouped_part_ids[i - 1] : comm_index_type{-1};
            auto cur_part = grouped_part_ids[i];
            auto next_part = grouped_part_ids[i + 1];  // safe due to sentinel
            if (cur_part != next_part) {
                sizes[cur_part] = grouped_range_ranks[i];
            }
            // write result shifted by one entry to get exclusive prefix sum
            ranks[orig_idxs[i]] = prev_part == cur_part
                                      ? grouped_range_ranks[i - 1]
                                      : LocalIndexType{};
        },
        num_ranges, range_sizes, tmp_part_ids, permutation, ranks, sizes);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES);


}  // namespace partition
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
