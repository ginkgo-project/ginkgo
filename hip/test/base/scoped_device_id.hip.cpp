// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
// prevent compilation failure related to disappearing assert(...) statements
#include <hip/hip_runtime.h>
// force-top: off


#include <gtest/gtest.h>


#include <ginkgo/core/base/executor.hpp>


#include "hip/base/scoped_device_id.hip.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


class ScopedDeviceIdGuard : public HipTestFixture {};


TEST_F(ScopedDeviceIdGuard, SetsId)
{
    auto new_device_id = std::max(exec->get_num_devices() - 1, 0);

    gko::detail::hip_scoped_device_id_guard g{new_device_id};

    int device_id;
    hipGetDevice(&device_id);
    ASSERT_EQ(device_id, new_device_id);
}


TEST_F(ScopedDeviceIdGuard, ResetsId)
{
    auto old_device_id = exec->get_device_id();

    {
        auto new_device_id = std::max(exec->get_num_devices() - 1, 0);
        gko::detail::hip_scoped_device_id_guard g{new_device_id};
    }

    int device_id;
    hipGetDevice(&device_id);
    ASSERT_EQ(device_id, old_device_id);
}


}  // namespace
