
#include "core/distributed/partition_helpers_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace partition_helpers {


template <typename GlobalIndexType>
void compress_start_ends(std::shared_ptr<const DefaultExecutor> exec,
                         const array<GlobalIndexType>& range_start_ends,
                         array<GlobalIndexType>& ranges)
{
    if (ranges.get_num_elems()) {
        ranges.get_data()[0] = range_start_ends.get_const_data()[0];
        for (size_type i = 0; i < ranges.get_num_elems() - 1; ++i) {
            ranges.get_data()[i + 1] =
                range_start_ends.get_const_data()[2 * i + 1];
        }
    }
}


GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_START_ENDS);


}  // namespace partition_helpers
}  // namespace reference
}  // namespace kernels
}  // namespace gko
