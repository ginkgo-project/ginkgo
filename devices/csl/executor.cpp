// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/executor.hpp"


namespace gko {


std::shared_ptr<Executor> CslExecutor::get_master() noexcept { return master_; }


std::shared_ptr<const Executor> CslExecutor::get_master() const noexcept
{
    return master_;
}


}  // namespace gko
