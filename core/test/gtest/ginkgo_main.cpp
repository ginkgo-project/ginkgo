// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <gtest/gtest.h>


#include "core/test/gtest/environments.hpp"


int ResourceEnvironment::omp_threads = 0;
int ResourceEnvironment::cuda_device_id = 0;
int ResourceEnvironment::hip_device_id = 0;
int ResourceEnvironment::sycl_device_id = 0;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new ResourceEnvironment);
    ::testing::AddGlobalTestEnvironment(new DeviceEnvironment(0));
    int result = RUN_ALL_TESTS();
    return result;
}
