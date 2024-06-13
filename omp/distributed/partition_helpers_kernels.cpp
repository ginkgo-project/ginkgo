// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_helpers_kernels.hpp"


#include "core/base/iterator_factory.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace partition_helpers {


template <typename GlobalIndexType>
void sort_by_range_start(
    std::shared_ptr<const DefaultExecutor> exec,
    array<GlobalIndexType>& range_start_ends,
    array<experimental::distributed::comm_index_type>& part_ids)
{
    auto part_ids_d = part_ids.get_data();
    auto num_parts = part_ids.get_size();
    auto start_it = detail::make_permute_iterator(
        range_start_ends.get_data(), [](const auto i) { return 2 * i; });
    auto end_it = detail::make_permute_iterator(
        range_start_ends.get_data() + 1, [](const auto i) { return 2 * i; });
    auto sort_it = detail::make_zip_iterator(start_it, end_it, part_ids_d);
    // TODO: use TBB or parallel std with c++17
    std::stable_sort(sort_it, sort_it + num_parts,
                     [](const auto& a, const auto& b) {
                         return std::get<0>(a) < std::get<0>(b);
                     });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


}  // namespace partition_helpers
}  // namespace omp
}  // namespace kernels
}  // namespace gko
