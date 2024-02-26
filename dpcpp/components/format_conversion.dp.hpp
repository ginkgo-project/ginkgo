// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
    size_type nsgs_in_dpcpp = exec->get_num_subgroups();
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
    return std::min(multiple * nsgs_in_dpcpp,
                    size_type(ceildiv(nnz, subgroup_size)));
}


}  // namespace host_kernel
}  // namespace coo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_FORMAT_CONVERSION_DP_HPP_
