// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename GlobalIndexType>
void sort_by_range_start(
    std::shared_ptr<const DefaultExecutor> exec,
    array<GlobalIndexType>& range_start_ends,
    array<experimental::distributed::comm_index_type>& part_ids)
{
    auto num_ranges = range_start_ends.get_size() / 2;
    auto strided_indices = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [] __host__ __device__(const int i) { return 2 * i; });
    auto start_it = thrust::make_permutation_iterator(
        range_start_ends.get_data(), strided_indices);
    auto end_it = thrust::make_permutation_iterator(
        range_start_ends.get_data() + 1, strided_indices);
    auto zip_it = thrust::make_zip_iterator(
        thrust::make_tuple(end_it, part_ids.get_data()));
    thrust::stable_sort_by_key(thrust_policy(exec), start_it,
                               start_it + num_ranges, zip_it);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);
