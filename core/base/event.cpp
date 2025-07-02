// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/event.hpp>
#include <ginkgo/core/base/executor.hpp>

namespace gko {


NotAsyncEvent::NotAsyncEvent(std::shared_ptr<const Executor> exec)
{
    exec->synchronize();
}


}  // namespace gko
