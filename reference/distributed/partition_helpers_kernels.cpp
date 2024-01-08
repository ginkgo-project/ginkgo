// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_helpers_kernels.hpp"


#include "core/base/iterator_factory.hpp"


namespace gko {
namespace kernels {
namespace reference {
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
    std::stable_sort(sort_it, sort_it + num_parts,
                     [](const auto& a, const auto& b) {
                         return std::get<0>(a) < std::get<0>(b);
                     });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


template <typename GlobalIndexType>
void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec,
                              const array<GlobalIndexType>& range_start_ends,
                              bool& result)
{
    auto num_parts = range_start_ends.get_size() / 2;
    auto start_it =
        detail::make_permute_iterator(range_start_ends.get_const_data() + 2,
                                      [](const auto i) { return 2 * i; });
    auto end_it =
        detail::make_permute_iterator(range_start_ends.get_const_data() + 1,
                                      [](const auto i) { return 2 * i; });
    auto range_it = detail::make_zip_iterator(start_it, end_it);

    if (num_parts) {
        result = std::all_of(
            range_it, range_it + num_parts - 1,
            [](const auto& r) { return std::get<0>(r) == std::get<1>(r); });
    } else {
        result = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES);


template <typename GlobalIndexType>
void compress_ranges(std::shared_ptr<const DefaultExecutor> exec,
                     const array<GlobalIndexType>& range_start_ends,
                     array<GlobalIndexType>& range_offsets)
{
    range_offsets.get_data()[0] = range_start_ends.get_const_data()[0];
    for (int i = 0; i < range_offsets.get_size() - 1; ++i) {
        range_offsets.get_data()[i + 1] =
            range_start_ends.get_const_data()[2 * i + 1];
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_RANGES);


}  // namespace partition_helpers
}  // namespace reference
}  // namespace kernels
}  // namespace gko
