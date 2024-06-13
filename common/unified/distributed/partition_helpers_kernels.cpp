// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_helpers_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"
#include "core/base/array_access.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition_helpers {


template <typename GlobalIndexType>
void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec,
                              const array<GlobalIndexType>& range_start_ends,
                              bool& result)
{
    array<uint32> result_uint32{exec, 1};
    auto num_ranges = range_start_ends.get_size() / 2;
    // need additional guard because DPCPP doesn't return the initial value for
    // empty inputs
    if (num_ranges > 1) {
        run_kernel_reduction(
            exec,
            [] GKO_KERNEL(const auto i, const auto* ranges) {
                return ranges[2 * i] == ranges[2 * i + 1];
            },
            [] GKO_KERNEL(const auto a, const auto b) {
                return static_cast<uint32>(a && b);
            },
            [] GKO_KERNEL(auto x) { return x; }, static_cast<uint32>(true),
            result_uint32.get_data(), num_ranges - 1,
            range_start_ends.get_const_data() + 1);
        result = static_cast<bool>(get_element(result_uint32, 0));
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
    run_kernel(
        exec,
        [] GKO_KERNEL(const auto i, const auto* start_ends, auto* offsets) {
            if (i == 0) {
                offsets[0] = start_ends[0];
            }
            offsets[i + 1] = start_ends[2 * i + 1];
        },
        range_offsets.get_size() - 1, range_start_ends.get_const_data(),
        range_offsets.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_RANGES);


}  // namespace partition_helpers
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
