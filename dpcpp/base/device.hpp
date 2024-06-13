// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_DEVICE_HPP_
#define GKO_DPCPP_BASE_DEVICE_HPP_


#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {


/** calls delete on the given event. */
void destroy_event(sycl::event* event);


std::string get_device_name(int device_id);


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_DEVICE_HPP_
