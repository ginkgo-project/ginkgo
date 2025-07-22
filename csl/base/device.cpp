// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "csl/base/device.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace csl {


void destroy_event(::csl::event* event) { delete event; }


std::string get_device_name(int device_id) { return "Csl device"; }


}  // namespace csl
}  // namespace kernels
}  // namespace gko
