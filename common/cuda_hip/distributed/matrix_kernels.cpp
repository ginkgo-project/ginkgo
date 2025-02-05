// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
#include "common/cuda_hip/components/searching.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace distributed_matrix {


template <typename ValueType, typename GlobalIndexType>
struct input_type {
    GlobalIndexType row;
    GlobalIndexType col;
    ValueType val;

    __forceinline__ __device__ __host__
    input_type(thrust::tuple<GlobalIndexType, GlobalIndexType, ValueType> t)
        : row(thrust::get<0>(t)), col(thrust::get<1>(t)), val(thrust::get<2>(t))
    {}
};


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
    auto map_to_row_range_id =
        [row_range_bounds, num_row_ranges] __host__ __device__(
            GlobalIndexType row) {
            return binary_search(size_type(0), num_row_ranges, [=](auto mid) {
                return row < row_range_bounds[mid + 1];
            });
        };
    auto map_to_col_range_id =
        [col_range_bounds, num_col_ranges] __host__ __device__(
            GlobalIndexType col) {
            return binary_search(size_type(0), num_col_ranges, [=](auto mid) {
                return col < col_range_bounds[mid + 1];
            });
        };

    // count number of local<0> and non-local<1> elements. Since the input
    // may contain non-local rows, we don't have
    // num_local + num_non_local = num_elements and can't just count one of them
    auto input_row_idxs = input.get_const_row_idxs();
    auto input_col_idxs = input.get_const_col_idxs();
    auto input_idxs_it = thrust::make_zip_iterator(
        thrust::make_tuple(input_row_idxs, input_col_idxs));
    auto num_elements_pair = thrust::transform_reduce(
        policy, input_idxs_it, input_idxs_it + num_input_elements,
        [local_part, row_part_ids, col_part_ids, map_to_row_range_id,
         map_to_col_range_id] __host__
            __device__(
                const thrust::tuple<GlobalIndexType, GlobalIndexType>& tuple) {
                auto row = thrust::get<0>(tuple);
                auto col = thrust::get<1>(tuple);
                auto row_part = row_part_ids[map_to_row_range_id(row)];
                auto col_part = col_part_ids[map_to_col_range_id(col)];
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
    auto map_to_local_row = [row_range_bounds, row_range_starting_indices,
                             map_to_row_range_id] __host__
                            __device__(const GlobalIndexType row) {
                                auto range_id = map_to_row_range_id(row);
                                return static_cast<LocalIndexType>(
                                           row - row_range_bounds[range_id]) +
                                       row_range_starting_indices[range_id];
                            };
    auto map_to_local_col = [col_range_bounds, col_range_starting_indices,
                             map_to_col_range_id] __host__
                            __device__(const GlobalIndexType col) {
                                auto range_id = map_to_col_range_id(col);
                                return static_cast<LocalIndexType>(
                                           col - col_range_bounds[range_id]) +
                                       col_range_starting_indices[range_id];
                            };

    using input_type = input_type<ValueType, GlobalIndexType>;
    auto input_it = thrust::make_zip_iterator(thrust::make_tuple(
        input.get_const_row_idxs(), input.get_const_col_idxs(),
        input.get_const_values()));

    // copy and transform local entries into arrays
    local_row_idxs.resize_and_reset(num_local_elements);
    local_col_idxs.resize_and_reset(num_local_elements);
    local_values.resize_and_reset(num_local_elements);
    auto local_it = thrust::make_transform_iterator(
        input_it, [map_to_local_row, map_to_local_col] __host__ __device__(
                      const input_type input) {
            auto local_row = map_to_local_row(input.row);
            auto local_col = map_to_local_col(input.col);
            return thrust::make_tuple(local_row, local_col, input.val);
        });
    thrust::copy_if(
        policy, local_it, local_it + input.get_num_stored_elements(),
        input_idxs_it,
        thrust::make_zip_iterator(thrust::make_tuple(local_row_idxs.get_data(),
                                                     local_col_idxs.get_data(),
                                                     local_values.get_data())),
        [local_part, row_part_ids, col_part_ids, map_to_row_range_id,
         map_to_col_range_id] __host__
            __device__(
                const thrust::tuple<GlobalIndexType, GlobalIndexType>& tuple) {
                auto row = thrust::get<0>(tuple);
                auto col = thrust::get<1>(tuple);
                auto row_part = row_part_ids[map_to_row_range_id(row)];
                auto col_part = col_part_ids[map_to_col_range_id(col)];
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
            auto local_row = map_to_local_row(input.row);
            return thrust::make_tuple(local_row, input.col, input.val);
        });
    thrust::copy_if(
        policy, non_local_it, non_local_it + input.get_num_stored_elements(),
        input_idxs_it,
        thrust::make_zip_iterator(thrust::make_tuple(
            non_local_row_idxs.get_data(), non_local_col_idxs.get_data(),
            non_local_values.get_data())),
        [local_part, row_part_ids, col_part_ids, map_to_row_range_id,
         map_to_col_range_id] __host__
            __device__(
                const thrust::tuple<GlobalIndexType, GlobalIndexType>& tuple) {
                auto row = thrust::get<0>(tuple);
                auto col = thrust::get<1>(tuple);
                auto row_part = row_part_ids[map_to_row_range_id(row)];
                auto col_part = col_part_ids[map_to_col_range_id(col)];
                return row_part == local_part && col_part != local_part;
            });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE_BASE(
    GKO_DECLARE_SEPARATE_LOCAL_NONLOCAL);


}  // namespace distributed_matrix
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
