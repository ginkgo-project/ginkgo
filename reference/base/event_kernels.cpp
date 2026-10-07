// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
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
namespace reference {
namespace event {


void record_event(std::shared_ptr<const DefaultExecutor> exec,
                  std::shared_ptr<const detail::Event>& event)
{
    event = std::make_shared<detail::NotAsyncEvent>(exec);
}


void wait_event(std::shared_ptr<const DefaultExecutor> exec,
                const detail::Event* event)
{
    // the host executes in order, so it has no queue to defer the wait to
    event->synchronize();
}


}  // namespace event
}  // namespace reference
}  // namespace kernels
}  // namespace gko
