
#include "core/distributed/partition_helpers_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition_helpers {


template <typename GlobalIndexType>
void compress_start_ends(std::shared_ptr<const DefaultExecutor> exec,
                         const array<GlobalIndexType>& range_start_ends,
                         array<GlobalIndexType>& ranges)
{
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto size, const auto* range_start_ends,
                      auto* ranges) {
            if (i == 0) {
                ranges[0] = range_start_ends[0];
            }
            if (i != size - 1) {
                ranges[i + 1] = range_start_ends[2 * i + 1];
            }
        },
        ranges.get_num_elems() - 1, ranges.get_num_elems(),
        range_start_ends.get_const_data(), ranges.get_data());
}


template <typename GlobalIndexType>
void check_consecutive_ranges()
{}


GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_START_ENDS);


template <typename GlobalIndexType>
void sort_by_range_start(std::shared_ptr<const DefaultExecutor> exec,
                         array<GlobalIndexType>& range_start_ends,
                         array<experimental::distributed::comm_index_type>&
                             part_ids) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


}  // namespace partition_helpers
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
