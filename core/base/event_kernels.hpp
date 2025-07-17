// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_EVENT_KERNELS_HPP_
#define GKO_CORE_BASE_EVENT_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_EVENT_RECORD_EVENT                             \
    void record_event(std::shared_ptr<const DefaultExecutor> exec, \
                      std::shared_ptr<const Event>& event)


#define GKO_DECLARE_ALL_AS_TEMPLATES GKO_DECLARE_EVENT_RECORD_EVENT


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(event, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_BASE_EVENT_KERNELS_HPP_
