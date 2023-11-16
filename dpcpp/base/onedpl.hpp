// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_ONEDPL_HPP_
#define GKO_DPCPP_BASE_ONEDPL_HPP_


#include <oneapi/dpl/execution>

#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {


inline auto onedpl_policy(std::shared_ptr<const DpcppExecutor> exec)
{
    return oneapi::dpl::execution::make_device_policy(*exec->get_queue());
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_ONEDPL_HPP_
