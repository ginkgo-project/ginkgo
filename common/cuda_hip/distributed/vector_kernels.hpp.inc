// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename LocalIndexType, typename GlobalIndexType>
void build_local(
    std::shared_ptr<const DefaultExecutor> exec,
    const device_matrix_data<ValueType, GlobalIndexType>& input,
    const experimental::distributed::Partition<LocalIndexType, GlobalIndexType>*
        partition,
    comm_index_type local_part, matrix::Dense<ValueType>* local_mtx)
{
    const auto* range_bounds = partition->get_range_bounds();
    const auto* range_starting_indices =
        partition->get_range_starting_indices();
    const auto* part_ids = partition->get_part_ids();
    const auto num_ranges = partition->get_num_ranges();

    array<size_type> range_id{exec, input.get_num_stored_elements()};
    thrust::upper_bound(
        thrust_policy(exec), range_bounds + 1, range_bounds + num_ranges + 1,
        input.get_const_row_idxs(),
        input.get_const_row_idxs() + input.get_num_stored_elements(),
        range_id.get_data(), thrust::less<GlobalIndexType>());

    // write values with local rows into the local matrix at the correct index
    // this needs the following iterators:
    // - local_row_it: (global_row, range_id) -> local row index
    // - flat_idx_it: (local_row, col) -> flat index in local matrix values
    //                                    array
    // the flat_idx_it is used by the scatter_if as an index map for the values
    auto map_to_local_row =
        [range_bounds, range_starting_indices] __host__ __device__(
            const thrust::tuple<GlobalIndexType, size_type>& idx_range_id) {
            const auto idx = thrust::get<0>(idx_range_id);
            const auto rid = thrust::get<1>(idx_range_id);
            return static_cast<LocalIndexType>(idx - range_bounds[rid]) +
                   range_starting_indices[rid];
        };
    auto local_row_it = thrust::make_transform_iterator(
        thrust::make_zip_iterator(thrust::make_tuple(input.get_const_row_idxs(),
                                                     range_id.get_data())),
        map_to_local_row);

    auto stride = local_mtx->get_stride();
    auto map_to_flat_idx =
        [stride] __host__ __device__(
            const thrust::tuple<LocalIndexType, GlobalIndexType>& row_col) {
            return thrust::get<0>(row_col) * stride + thrust::get<1>(row_col);
        };
    auto flat_idx_it = thrust::make_transform_iterator(
        thrust::make_zip_iterator(
            thrust::make_tuple(local_row_it, input.get_const_col_idxs())),
        map_to_flat_idx);

    auto is_local_row =
        [part_ids, local_part] __host__ __device__(const size_type rid) {
            return part_ids[rid] == local_part;
        };
    thrust::scatter_if(
        thrust_policy(exec), input.get_const_values(),
        input.get_const_values() + input.get_num_stored_elements(), flat_idx_it,
        range_id.get_data(), local_mtx->get_values(), is_local_row);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_LOCAL_GLOBAL_INDEX_TYPE(
    GKO_DECLARE_DISTRIBUTED_VECTOR_BUILD_LOCAL);
