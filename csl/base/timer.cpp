// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/timer.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


CslTimer::CslTimer(std::shared_ptr<const CslExecutor> exec)
{
    // TODO
}


void CslTimer::init_time_point(time_point& time)
{
    // TODO
}


void CslTimer::record(time_point& time)
{
    // TODO
}


void CslTimer::wait(time_point& time)
{
    // TODO
}


std::chrono::nanoseconds CslTimer::difference_async(const time_point& start,
                                                    const time_point& stop)
{
    return std::chrono::nanoseconds{static_cast<int64>(0)};
}


}  // namespace gko
