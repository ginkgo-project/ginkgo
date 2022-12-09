
#include "core/distributed/partition_helpers_kernels.hpp"
#include <numeric>

#include "core/base/iterator_factory.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace partition_helpers {


template <typename GlobalIndexType>
void compress_start_ends(std::shared_ptr<const DefaultExecutor> exec,
                         const array<GlobalIndexType>& range_start_ends,
                         array<GlobalIndexType>& ranges)
{
    if (ranges.get_num_elems() && range_start_ends.get_num_elems()) {
        ranges.get_data()[0] = range_start_ends.get_const_data()[0];
        for (size_type i = 0; i < ranges.get_num_elems() - 1; ++i) {
            ranges.get_data()[i + 1] =
                range_start_ends.get_const_data()[2 * i + 1];
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_START_ENDS);


template <typename GlobalIndexType>
void sort_by_range_start(
    std::shared_ptr<const DefaultExecutor> exec,
    array<GlobalIndexType>& range_start_ends,
    array<experimental::distributed::comm_index_type>& part_ids)
{
    auto part_ids_d = part_ids.get_data();
    auto num_parts = part_ids.get_num_elems();
    auto range_starts = range_start_ends.get_data();
    auto range_ends = range_starts + num_parts;
    auto sort_it =
        detail::make_zip_iterator(range_starts, range_ends, part_ids_d);
    std::sort(sort_it, sort_it + num_parts, [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


template <typename GlobalIndexType>
void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec,
                              array<GlobalIndexType>& range_start_ends,
                              bool* result)
{
    auto num_parts = range_start_ends.get_num_elems() / 2;
    auto range_starts = range_start_ends.get_data();
    auto range_ends = range_starts + num_parts;
    auto combined_it = detail::make_zip_iterator(range_starts + 1, range_ends);

    if (num_parts) {
        *result = std::all_of(combined_it, combined_it + (num_parts - 1),
                              [](const auto& start_end) {
                                  return std::get<0>(start_end) ==
                                         std::get<1>(start_end);
                              });
    } else {
        *result = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES);


}  // namespace partition_helpers
}  // namespace reference
}  // namespace kernels
}  // namespace gko
