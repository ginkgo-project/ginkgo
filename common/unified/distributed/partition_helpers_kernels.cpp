
#include "core/distributed/partition_helpers_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition_helpers {


template <typename GlobalIndexType>
void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec,
                              array<GlobalIndexType>& range_start_ends,
                              bool* result)
{
    array<uint32> result_uint32{exec, 1};
    auto num_ranges = range_start_ends.get_num_elems() / 2;
    run_kernel_reduction(
        exec,
        [] GKO_KERNEL(const auto i, const auto* starts, const auto* ends) {
            return starts[i + 1] == ends[i];
        },
        [] GKO_KERNEL(const auto a, const auto b) {
            return static_cast<uint32>(a && b);
        },
        [] GKO_KERNEL(auto x) { return x; }, static_cast<uint32>(true),
        result_uint32.get_data(), num_ranges - 1, range_start_ends.get_data(),
        range_start_ends.get_data() + num_ranges);
    *result =
        static_cast<bool>(exec->copy_val_to_host(result_uint32.get_data()));
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_CHECK_CONSECUTIVE_RANGES);

}  // namespace partition_helpers
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
