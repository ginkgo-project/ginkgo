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


namespace gko {
namespace kernels {
namespace reference {
namespace partition {


void count_ranges(std::shared_ptr<const DefaultExecutor> exec,
                  const array<comm_index_type>& mapping, size_type& num_ranges)
{
    num_ranges = 0;
    comm_index_type prev_part{-1};
    for (size_type i = 0; i < mapping.get_num_elems(); i++) {
        auto cur_part = mapping.get_const_data()[i];
        num_ranges += cur_part != prev_part;
        prev_part = cur_part;
    }
}


template <typename GlobalIndexType>
void build_from_contiguous(std::shared_ptr<const DefaultExecutor> exec,
                           const array<GlobalIndexType>& ranges,
                           const array<comm_index_type>& part_id_mapping,
                           GlobalIndexType* range_bounds,
                           comm_index_type* part_ids)
{
    bool uses_mapping = part_id_mapping.get_num_elems() > 0;
    range_bounds[0] = 0;
    for (comm_index_type i = 0; i < ranges.get_num_elems() - 1; i++) {
        auto end = ranges.get_const_data()[i + 1];
        range_bounds[i + 1] = end;
        part_ids[i] = uses_mapping ? part_id_mapping.get_const_data()[i] : i;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_CONTIGUOUS);


template <typename GlobalIndexType>
void build_from_mapping(std::shared_ptr<const DefaultExecutor> exec,
                        const array<comm_index_type>& mapping,
                        GlobalIndexType* range_bounds,
                        comm_index_type* part_ids)
{
    size_type range_idx{};
    comm_index_type range_part{-1};
    for (size_type i = 0; i < mapping.get_num_elems(); i++) {
        auto cur_part = mapping.get_const_data()[i];
        if (cur_part != range_part) {
            range_bounds[range_idx] = i;
            part_ids[range_idx] = cur_part;
            range_idx++;
            range_part = cur_part;
        }
    }
    range_bounds[range_idx] =
        static_cast<GlobalIndexType>(mapping.get_num_elems());
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

    auto* ranges_ptr = ranges.get_data();

    ranges_ptr[0] = 0;
    for (int i = 1; i < num_parts + 1; ++i) {
        ranges_ptr[i] =
            ranges_ptr[i - 1] + size_per_part + ((i - 1) < rest ? 1 : 0);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_PARTITION_BUILD_FROM_GLOBAL_SIZE);


template <typename LocalIndexType, typename GlobalIndexType>
void build_starting_indices(std::shared_ptr<const DefaultExecutor> exec,
                            const GlobalIndexType* range_offsets,
                            const int* range_parts, size_type num_ranges,
                            int num_parts, int& num_empty_parts,
                            LocalIndexType* ranks, LocalIndexType* sizes)
{
    std::fill_n(sizes, num_parts, 0);
    for (size_type range = 0; range < num_ranges; ++range) {
        auto begin = range_offsets[range];
        auto end = range_offsets[range + 1];
        auto part = range_parts[range];
        auto rank = sizes[part];
        ranks[range] = rank;
        sizes[part] += end - begin;
    }
    num_empty_parts = std::count(sizes, sizes + num_parts, 0);
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES);

template <typename LocalIndexType, typename GlobalIndexType>
void has_ordered_parts(
    std::shared_ptr<const DefaultExecutor> exec,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    bool* result)
{
    *result = true;
    auto part_ids = partition->get_part_ids();

    for (comm_index_type i = 1; i < partition->get_num_ranges(); ++i) {
        if (part_ids[i] < part_ids[i - 1]) {
            *result = false;
            return;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PARTITION_IS_ORDERED);

}  // namespace partition
}  // namespace reference
}  // namespace kernels
}  // namespace gko
