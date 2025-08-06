// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_helpers_kernels.hpp"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>

#include "common/cuda_hip/base/thrust.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition_helpers {


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


}  // namespace partition_helpers
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
