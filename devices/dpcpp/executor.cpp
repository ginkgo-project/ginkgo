// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/executor.hpp>


#include <cstdlib>
#include <cstring>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


std::shared_ptr<Executor> DpcppExecutor::get_master() noexcept
{
    return master_;
}


std::shared_ptr<const Executor> DpcppExecutor::get_master() const noexcept
{
    return master_;
}


}  // namespace gko
