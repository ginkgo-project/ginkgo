// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CSL_BASE_DEVICE_HPP_
#define GKO_CSL_BASE_DEVICE_HPP_

#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace csl {


/** calls delete on the given event. */
void destroy_event(::csl::event* event);


std::string get_device_name(int device_id);


}  // namespace csl
}  // namespace kernels
}  // namespace gko

#endif  // GKO_CSL_BASE_DEVICE_HPP_
