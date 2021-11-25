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


#include "common/unified/distributed/partition_kernels.hpp.inc"


template <typename LocalIndexType, typename GlobalIndexType>
void build_starting_indices(std::shared_ptr<const DefaultExecutor> exec,
                            const GlobalIndexType* range_offsets,
                            const comm_index_type* range_parts,
                            size_type num_ranges, comm_index_type num_parts,
                            comm_index_type& num_empty_parts,
                            LocalIndexType* starting_indices,
                            LocalIndexType* part_sizes)
{
    if (num_ranges > 0) {
        Array<LocalIndexType> range_sizes{exec, num_ranges};
        // num_parts sentinel at the end
        Array<comm_index_type> tmp_part_ids{exec, num_ranges + 1};
        Array<GlobalIndexType> permutation{exec, num_ranges};
        // set part_sizes to 0 in case of empty parts
        components::fill_array(exec, part_sizes, num_parts, LocalIndexType{});

        kernel::setup_sizes_ids_permutation(
            exec, num_ranges, num_parts, range_offsets, range_parts,
            range_sizes, tmp_part_ids, permutation);

        auto tmp_part_id_ptr =
            thrust::device_pointer_cast(tmp_part_ids.get_data());
        auto range_sizes_ptr =
            thrust::device_pointer_cast(range_sizes.get_data());
        auto permutation_ptr =
            thrust::device_pointer_cast(permutation.get_data());
        auto value_it = thrust::make_zip_iterator(
            thrust::make_tuple(range_sizes_ptr, permutation_ptr));
        // group range_sizes by part ID
        thrust::stable_sort_by_key(thrust::device, tmp_part_id_ptr,
                                   tmp_part_id_ptr + num_ranges, value_it);
        // compute inclusive prefix sum for each part
        thrust::inclusive_scan_by_key(thrust::device, tmp_part_id_ptr,
                                      tmp_part_id_ptr + num_ranges,
                                      range_sizes_ptr, range_sizes_ptr);
        // write back the results
        kernel::compute_part_sizes_and_starting_indices(
            exec, num_ranges, range_sizes, tmp_part_ids, permutation,
            starting_indices, part_sizes);
        num_empty_parts = thrust::count(thrust::device, part_sizes,
                                        part_sizes + num_parts, 0);
    } else {
        num_empty_parts = num_parts;
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES);
