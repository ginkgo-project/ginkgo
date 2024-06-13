// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
