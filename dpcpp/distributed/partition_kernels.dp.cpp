// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/iterator>
// force-top: off


#include "core/distributed/partition_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "dpcpp/base/onedpl.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace partition {
namespace kernel {


template <typename LocalIndexType, typename GlobalIndexType>
void setup_sizes_ids_permutation(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_ranges,
    comm_index_type num_parts, const GlobalIndexType* range_offsets,
    const comm_index_type* range_parts, Array<LocalIndexType>& range_sizes,
    Array<comm_index_type>& part_ids, Array<GlobalIndexType>& permutation)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto num_ranges, auto num_parts,
                      auto range_offsets, auto range_parts, auto range_sizes,
                      auto part_ids, auto permutation) {
            if (i == 0) {
                // set sentinel value at the end
                part_ids[num_ranges] = num_parts;
            }
            range_sizes[i] = range_offsets[i + 1] - range_offsets[i];
            part_ids[i] = range_parts[i];
            permutation[i] = static_cast<GlobalIndexType>(i);
        },
        num_ranges, num_ranges, num_parts, range_offsets, range_parts,
        range_sizes, part_ids, permutation);
}


template <typename LocalIndexType, typename GlobalIndexType>
void compute_part_sizes_and_starting_indices(
    std::shared_ptr<const DefaultExecutor> exec, size_type num_ranges,
    const Array<LocalIndexType>& range_sizes,
    const Array<comm_index_type>& part_ids,
    const Array<GlobalIndexType>& permutation, LocalIndexType* starting_indices,
    LocalIndexType* part_sizes)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto grouped_starting_indices,
                      auto grouped_part_ids, auto orig_idxs,
                      auto starting_indices, auto part_sizes) {
            auto prev_part = i > 0 ? grouped_part_ids[i - 1]
                                   : invalid_index<comm_index_type>();
            auto cur_part = grouped_part_ids[i];
            auto next_part =
                grouped_part_ids[i + 1];  // last element has to be num_parts
            if (cur_part != next_part) {
                part_sizes[cur_part] = grouped_starting_indices[i];
            }
            // write result shifted by one entry to get exclusive prefix sum
            starting_indices[orig_idxs[i]] =
                prev_part == cur_part ? grouped_starting_indices[i - 1]
                                      : LocalIndexType{};
        },
        num_ranges, range_sizes, part_ids, permutation, starting_indices,
        part_sizes);
}


}  // namespace kernel


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
        auto policy = onedpl_policy(exec);

        Array<LocalIndexType> range_sizes{exec, num_ranges};
        // num_parts sentinel at the end
        Array<comm_index_type> tmp_part_ids{exec, num_ranges + 1};
        Array<GlobalIndexType> permutation{exec, num_ranges};
        // set part_sizes to 0 in case of empty parts
        components::fill_array(exec, part_sizes, num_parts,
                               zero<LocalIndexType>());

        kernel::setup_sizes_ids_permutation(
            exec, num_ranges, num_parts, range_offsets, range_parts,
            range_sizes, tmp_part_ids, permutation);

        auto tmp_part_id_ptr = tmp_part_ids.get_data();
        auto range_sizes_ptr = range_sizes.get_data();
        auto sort_it = oneapi::dpl::make_zip_iterator(
            tmp_part_id_ptr, range_sizes_ptr, permutation.get_data());
        // group range_sizes by part ID
        oneapi::dpl::stable_sort(policy, sort_it, sort_it + num_ranges,
                                 [](const auto t_a, const auto t_b) {
                                     return std::get<0>(t_a) < std::get<0>(t_b);
                                 });
        // compute inclusive prefix sum for each part
        oneapi::dpl::inclusive_scan_by_segment(
            policy, tmp_part_id_ptr, tmp_part_id_ptr + num_ranges,
            range_sizes_ptr, range_sizes_ptr);
        // write back the results
        kernel::compute_part_sizes_and_starting_indices(
            exec, num_ranges, range_sizes, tmp_part_ids, permutation,
            starting_indices, part_sizes);
        num_empty_parts =
            oneapi::dpl::count(policy, part_sizes, part_sizes + num_parts, 0);
    } else {
        num_empty_parts = num_parts;
    }
}

GKO_INSTANTIATE_FOR_EACH_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_PARTITION_BUILD_STARTING_INDICES);


}  // namespace partition
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
