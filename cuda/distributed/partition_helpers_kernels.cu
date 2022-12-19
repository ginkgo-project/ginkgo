
#include "core/distributed/partition_helpers_kernels.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace partition_helpers {


template <typename GlobalIndexType>
void sort_by_range_start(std::shared_ptr<const DefaultExecutor> exec,
                         array<GlobalIndexType>& range_start_ends,
                         array<experimental::distributed::comm_index_type>&
                             part_ids) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_SORT_BY_RANGE_START);


}  // namespace partition_helpers
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
