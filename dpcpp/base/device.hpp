// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_DEVICE_HPP_
#define GKO_DPCPP_BASE_DEVICE_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/export_dpcpp.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {


/** calls delete on the given event. */
GKO_DPCPP_EXPORT void destroy_event(sycl::event* event);


GKO_DPCPP_EXPORT std::string get_device_name(int device_id);


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_BASE_DEVICE_HPP_
