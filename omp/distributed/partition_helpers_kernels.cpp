
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
    auto num_parts = part_ids.get_num_elems();
    auto range_starts = range_start_ends.get_data();
    auto range_ends = range_starts + num_parts;
    auto sort_it =
        detail::make_zip_iterator(range_starts, range_ends, part_ids_d);
    // TODO: use TBB or parallel std with c++17
    std::sort(sort_it, sort_it + num_parts, [](const auto& a, const auto& b) {
        return std::get<0>(a) < std::get<0>(b);
    });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


}  // namespace partition_helpers
}  // namespace omp
}  // namespace kernels
}  // namespace gko
