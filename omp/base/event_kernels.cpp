// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/event_kernels.hpp"

#include <memory>

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "core/base/event.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace event {


void record_event(std::shared_ptr<const DefaultExecutor> exec,
                  std::shared_ptr<const detail::Event>& event)
{
    event = std::make_shared<detail::NotAsyncEvent>(exec);
}


}  // namespace event
}  // namespace omp
}  // namespace kernels
}  // namespace gko
