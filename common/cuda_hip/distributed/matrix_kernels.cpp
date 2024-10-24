// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/matrix_kernels.hpp"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/atomic.hpp"
#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace distributed_matrix {


template <typename ValueType, typename GlobalIndexType>
struct input_type {
    GlobalIndexType row;
    GlobalIndexType col;
    ValueType val;
    size_type row_range;
    size_type col_range;

    __forceinline__ __device__ __host__
    input_type(thrust::tuple<GlobalIndexType, GlobalIndexType, ValueType,
                             size_type, size_type>
                   t)
        : row(thrust::get<0>(t)),
          col(thrust::get<1>(t)),
          val(thrust::get<2>(t)),
          row_range(thrust::get<3>(t)),
          col_range(thrust::get<4>(t))
    {}
};


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
    components::fill_array(exec, row_part_ptrs.get_data(), num_parts + 1,
                           zero<comm_index_type>());

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


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void fill_send_buffers(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    comm_index_type local_part, const array<GlobalIndexType>& send_positions,
    const array<GlobalIndexType>& original_positions,
    array<GlobalIndexType>& send_row_idxs,
    array<GlobalIndexType>& send_col_idxs, array<ValueType>& send_values)
{
    auto num_entries = input.get_num_stored_elements();
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_values = input.get_const_values();

    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in_rows, auto in_cols, auto in_vals,
                      auto in_pos, auto out_pos, auto out_rows, auto out_cols,
                      auto out_vals) {
            if (in_pos[i] >= 0) {
                out_rows[out_pos[i]] = in_rows[in_pos[i]];
                out_cols[out_pos[i]] = in_cols[in_pos[i]];
                out_vals[out_pos[i]] = in_vals[in_pos[i]];
            }
        },
        num_entries, input_row_idxs, input_col_idxs, input_values,
        original_positions.get_const_data(), send_positions.get_const_data(),
        send_row_idxs.get_data(), send_col_idxs.get_data(),
        send_values.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILL_SEND_BUFFERS);


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void separate_local_nonlocal(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    experimental::distributed::comm_index_type local_part,
    array<LocalIndexType>& local_row_idxs,
    array<LocalIndexType>& local_col_idxs, array<ValueType>& local_values,
    array<LocalIndexType>& non_local_row_idxs,
    array<GlobalIndexType>& non_local_col_idxs,
    array<ValueType>& non_local_values)
{
    auto input_vals = input.get_const_values();
    auto row_part_ids = row_partition->get_part_ids();
    auto col_part_ids = col_partition->get_part_ids();
    const auto* row_range_bounds = row_partition->get_range_bounds();
    const auto* col_range_bounds = col_partition->get_range_bounds();
    const auto* row_range_starting_indices =
        row_partition->get_range_starting_indices();
    const auto* col_range_starting_indices =
        col_partition->get_range_starting_indices();
    const auto num_row_ranges = row_partition->get_num_ranges();
    const auto num_col_ranges = col_partition->get_num_ranges();
    const auto num_input_elements = input.get_num_stored_elements();

    auto policy = thrust_policy(exec);

    // precompute the row and column range id of each input element
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    array<size_type> row_range_ids{exec, num_input_elements};
    thrust::upper_bound(policy, row_range_bounds + 1,
                        row_range_bounds + num_row_ranges + 1, input_row_idxs,
                        input_row_idxs + num_input_elements,
                        row_range_ids.get_data());
    array<size_type> col_range_ids{exec, input.get_num_stored_elements()};
    thrust::upper_bound(policy, col_range_bounds + 1,
                        col_range_bounds + num_col_ranges + 1, input_col_idxs,
                        input_col_idxs + num_input_elements,
                        col_range_ids.get_data());

    // count number of local<0> and non-local<1> elements. Since the input
    // may contain non-local rows, we don't have
    // num_local + num_non_local = num_elements and can't just count one of them
    auto range_ids_it = thrust::make_zip_iterator(thrust::make_tuple(
        row_range_ids.get_const_data(), col_range_ids.get_const_data()));
    auto num_elements_pair = thrust::transform_reduce(
        policy, range_ids_it, range_ids_it + num_input_elements,
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            bool is_inner_entry =
                row_part == local_part && col_part == local_part;
            bool is_ghost_entry =
                row_part == local_part && col_part != local_part;
            return thrust::make_tuple(
                is_inner_entry ? size_type{1} : size_type{0},
                is_ghost_entry ? size_type{1} : size_type{0});
        },
        thrust::make_tuple(size_type{}, size_type{}),
        [] __host__ __device__(const thrust::tuple<size_type, size_type>& a,
                               const thrust::tuple<size_type, size_type>& b) {
            return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                      thrust::get<1>(a) + thrust::get<1>(b));
        });
    auto num_local_elements = thrust::get<0>(num_elements_pair);
    auto num_non_local_elements = thrust::get<1>(num_elements_pair);

    // define global-to-local maps for row and column indices
    auto map_to_local_row =
        [row_range_bounds, row_range_starting_indices] __host__ __device__(
            const GlobalIndexType row, const size_type range_id) {
            return static_cast<LocalIndexType>(row -
                                               row_range_bounds[range_id]) +
                   row_range_starting_indices[range_id];
        };
    auto map_to_local_col =
        [col_range_bounds, col_range_starting_indices] __host__ __device__(
            const GlobalIndexType col, const size_type range_id) {
            return static_cast<LocalIndexType>(col -
                                               col_range_bounds[range_id]) +
                   col_range_starting_indices[range_id];
        };

    using input_type = input_type<ValueType, GlobalIndexType>;
    auto input_it = thrust::make_zip_iterator(thrust::make_tuple(
        input.get_const_row_idxs(), input.get_const_col_idxs(),
        input.get_const_values(), row_range_ids.get_const_data(),
        col_range_ids.get_const_data()));

    // copy and transform local entries into arrays
    local_row_idxs.resize_and_reset(num_local_elements);
    local_col_idxs.resize_and_reset(num_local_elements);
    local_values.resize_and_reset(num_local_elements);
    auto local_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row, map_to_local_col] __host__ __device__(
                      const input_type input) {
            auto local_row = map_to_local_row(input.row, input.row_range);
            auto local_col = map_to_local_col(input.col, input.col_range);
            return thrust::make_tuple(local_row, local_col, input.val);
        });
    thrust::copy_if(
        policy, local_it, local_it + input.get_num_stored_elements(),
        range_ids_it,
        thrust::make_zip_iterator(thrust::make_tuple(local_row_idxs.get_data(),
                                                     local_col_idxs.get_data(),
                                                     local_values.get_data())),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part == local_part;
        });


    // copy and transform non-local entries into arrays. this keeps global
    // column indices, and also stores the column part id for each non-local
    // entry in an array
    non_local_row_idxs.resize_and_reset(num_non_local_elements);
    non_local_col_idxs.resize_and_reset(num_non_local_elements);
    non_local_values.resize_and_reset(num_non_local_elements);
    auto non_local_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row,
                   col_part_ids] __host__ __device__(const input_type input) {
            auto local_row = map_to_local_row(input.row, input.row_range);
            return thrust::make_tuple(local_row, input.col, input.val);
        });
    thrust::copy_if(
        policy, non_local_it, non_local_it + input.get_num_stored_elements(),
        range_ids_it,
        thrust::make_zip_iterator(thrust::make_tuple(
            non_local_row_idxs.get_data(), non_local_col_idxs.get_data(),
            non_local_values.get_data())),
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            return row_part == local_part && col_part != local_part;
        });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_SEPARATE_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
