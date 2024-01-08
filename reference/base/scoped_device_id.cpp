// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/scoped_device_id_guard.hpp>


#include "core/base/noop_scoped_device_id_guard.hpp"


namespace gko {


scoped_device_id_guard::scoped_device_id_guard(const ReferenceExecutor* exec,
                                               int device_id)
    : scope_(std::make_unique<detail::noop_scoped_device_id_guard>())
{}


}  // namespace gko
