/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/


#include "core/distributed/partition_helpers_kernels.hpp"


#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace partition_helpers {


template <typename GlobalIndexType>
void check_consecutive_ranges(std::shared_ptr<const DefaultExecutor> exec,
                              const array<GlobalIndexType>& range_start_ends,
                              bool* result)
{
    array<uint32> result_uint32{exec, 1};
    auto num_ranges = range_start_ends.get_num_elems() / 2;
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
        *result =
            static_cast<bool>(exec->copy_val_to_host(result_uint32.get_data()));
    } else {
        *result = true;
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
        range_offsets.get_num_elems() - 1, range_start_ends.get_const_data(),
        range_offsets.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_PARTITION_HELPERS_COMPRESS_RANGES);


}  // namespace partition_helpers
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
