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

#ifndef GKO_DPCPP_COMPONENTS_FORMAT_CONVERSION_DP_HPP_
#define GKO_DPCPP_COMPONENTS_FORMAT_CONVERSION_DP_HPP_


#include <algorithm>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
#include "benchmark/utils/tuning_variables.hpp"
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING


namespace gko {
namespace kernels {
namespace dpcpp {
namespace coo {
namespace host_kernel {


/**
 * @internal
 *
 * It calculates the number of warps used in Coo Spmv depending on the GPU
 * architecture and the number of stored elements.
 */
template <size_type subgroup_size = config::warp_size>
size_type calculate_nwarps(std::shared_ptr<const DpcppExecutor> exec,
                           const size_type nnz)
{
    size_type nwarps_in_dpcpp = exec->get_num_computing_units() * 8;
    size_type multiple = 8;
    if (nnz >= 2e8) {
        multiple = 256;
    } else if (nnz >= 2e7) {
        multiple = 32;
    }
#ifdef GINKGO_BENCHMARK_ENABLE_TUNING
    if (_tuning_flag) {
        multiple = _tuned_value;
    }
#endif  // GINKGO_BENCHMARK_ENABLE_TUNING
    return std::min(multiple * nwarps_in_dpcpp,
                    size_type(ceildiv(nnz, subgroup_size)));
}


}  // namespace host_kernel
}  // namespace coo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_FORMAT_CONVERSION_DP_HPP_
