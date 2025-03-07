// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace csl {


/** calls delete on the given event. */
void destroy_event(csl::event* event);


std::string get_device_name(int device_id);


}  // namespace csl
}  // namespace kernels
}  // namespace gko
