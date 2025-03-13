// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/dd_matrix_kernels.hpp"

#include <iostream>
#include <iterator>

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


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace distributed_dd_matrix {


template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void filter_non_owning_idxs(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        row_partition,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        col_partition,
    comm_index_type local_part, array<GlobalIndexType>& non_local_row_idxs,
    array<GlobalIndexType>& non_local_col_idxs)
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

    // count number of non local row and column indices.
    auto range_ids_it = thrust::make_zip_iterator(thrust::make_tuple(
        row_range_ids.get_const_data(), col_range_ids.get_const_data()));
    auto num_elements_pair = thrust::transform_reduce(
        policy, range_ids_it, range_ids_it + num_input_elements,
        [local_part, row_part_ids, col_part_ids] __host__ __device__(
            const thrust::tuple<size_type, size_type>& tuple) {
            auto row_part = row_part_ids[thrust::get<0>(tuple)];
            auto col_part = col_part_ids[thrust::get<1>(tuple)];
            bool is_local_row = row_part == local_part;
            bool is_local_col = col_part == local_part;
            return thrust::make_tuple(
                is_local_row ? size_type{0} : size_type{1},
                is_local_col ? size_type{0} : size_type{1});
        },
        thrust::make_tuple(size_type{}, size_type{}),
        [] __host__ __device__(const thrust::tuple<size_type, size_type>& a,
                               const thrust::tuple<size_type, size_type>& b) {
            return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                      thrust::get<1>(a) + thrust::get<1>(b));
        });
    auto n_non_local_row_idxs = thrust::get<0>(num_elements_pair);
    auto n_non_local_col_idxs = thrust::get<1>(num_elements_pair);

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

    non_local_col_idxs.resize_and_reset(n_non_local_col_idxs);
    non_local_row_idxs.resize_and_reset(n_non_local_row_idxs);
    thrust::copy_if(policy, input_col_idxs, input_col_idxs + num_input_elements,
                    range_ids_it, non_local_col_idxs.get_data(),
                    [local_part, col_part_ids] __host__ __device__(
                        const thrust::tuple<size_type, size_type>& tuple) {
                        auto col_part = col_part_ids[thrust::get<1>(tuple)];
                        return col_part != local_part;
                    });
    thrust::copy_if(policy, input_row_idxs, input_row_idxs + num_input_elements,
                    range_ids_it, non_local_row_idxs.get_data(),
                    [local_part, row_part_ids] __host__ __device__(
                        const thrust::tuple<size_type, size_type>& tuple) {
                        auto row_part = row_part_ids[thrust::get<0>(tuple)];
                        return row_part != local_part;
                    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_FILTER_NON_OWNING_IDXS);


}  // namespace distributed_dd_matrix
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
