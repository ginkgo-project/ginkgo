// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/base/executor.hpp>


namespace gko {
namespace kernels {
namespace poplar {


/** calls delete on the given event. */
void destroy_event(poplar::event* event);


std::string get_device_name(int device_id);


}  // namespace poplar
}  // namespace kernels
}  // namespace gko
