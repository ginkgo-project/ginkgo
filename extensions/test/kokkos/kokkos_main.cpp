// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <Kokkos_Core.hpp>


#include <gtest/gtest.h>


#include "core/test/gtest/environments.hpp"


int get_device_id()
{
#if defined(KOKKOS_ENABLE_CUDA)
    return ResourceEnvironment::cuda_device_id;
#elif defined(KOKKOS_ENABLE_HIP)
    return ResourceEnvironment::hip_device_id;
#elif defined(KOKKOS_ENABLE_SYCL)
    return ResourceEnvironment::sycl_device_id;
#else
    return 0;
#endif
}


class KokkosEnvironment : public ::testing::Environment {
public:
    void SetUp() override
    {
        Kokkos::initialize(
            Kokkos::InitializationSettings().set_device_id(get_device_id()));
    }

    void TearDown() override { Kokkos::finalize(); }
};


int ResourceEnvironment::omp_threads = 0;
int ResourceEnvironment::cuda_device_id = 0;
int ResourceEnvironment::hip_device_id = 0;
int ResourceEnvironment::sycl_device_id = 0;


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    ::testing::AddGlobalTestEnvironment(new ResourceEnvironment);
    ::testing::AddGlobalTestEnvironment(new DeviceEnvironment(0));
    ::testing::AddGlobalTestEnvironment(new KokkosEnvironment);
    int result = RUN_ALL_TESTS();
    return result;
}
