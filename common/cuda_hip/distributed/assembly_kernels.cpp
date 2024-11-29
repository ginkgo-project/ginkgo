// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/assembly_kernels.hpp"

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/thrust.hpp"
#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace assembly {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void count_non_owning_entries(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    comm_index_type local_part, array<comm_index_type>& send_count,
    array<GlobalIndexType>& send_positions,
    array<GlobalIndexType>& original_positions)
{
    auto row_part_ids = row_partition->get_part_ids();
    const auto* row_range_bounds = row_partition->get_range_bounds();
    const auto* row_range_starting_indices =
        row_partition->get_range_starting_indices();
    const auto num_row_ranges = row_partition->get_num_ranges();
    const auto num_input_elements = input.get_num_stored_elements();

    auto policy = thrust_policy(exec);

    // precompute the row and column range id of each input element
    auto input_row_idxs = input.get_const_row_idxs();
    array<size_type> row_range_ids{exec, num_input_elements};
    thrust::upper_bound(policy, row_range_bounds + 1,
                        row_range_bounds + num_row_ranges + 1, input_row_idxs,
                        input_row_idxs + num_input_elements,
                        row_range_ids.get_data());

    array<comm_index_type> row_part_ids_per_entry{exec, num_input_elements};
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto part_id, auto part_ids, auto range_ids,
                      auto part_ids_per_entry, auto orig_positions) {
            part_ids_per_entry[i] = part_ids[range_ids[i]];
            orig_positions[i] = part_ids_per_entry[i] == part_id ? -1 : i;
        },
        num_input_elements, local_part, row_part_ids, row_range_ids.get_data(),
        row_part_ids_per_entry.get_data(), original_positions.get_data());

    thrust::stable_sort_by_key(
        policy, row_part_ids_per_entry.get_data(),
        row_part_ids_per_entry.get_data() + num_input_elements,
        original_positions.get_data());
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto orig_positions, auto s_positions) {
            s_positions[i] = orig_positions[i] >= 0 ? 1 : 0;
        },
        num_input_elements, original_positions.get_const_data(),
        send_positions.get_data());

    components::prefix_sum_nonnegative(exec, send_positions.get_data(),
                                       num_input_elements);
    size_type num_parts = row_partition->get_num_parts();
    array<comm_index_type> row_part_ptrs{exec, num_parts + 1};

    components::convert_idxs_to_ptrs(
        exec, row_part_ids_per_entry.get_const_data(), num_input_elements,
        num_parts, row_part_ptrs.get_data());

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto part_id, auto part_ptrs, auto count) {
            count[i] = i == part_id ? 0 : part_ptrs[i + 1] - part_ptrs[i];
        },
        num_parts, local_part, row_part_ptrs.get_data(), send_count.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_COUNT_NON_OWNING_ENTRIES);


}  // namespace assembly
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
