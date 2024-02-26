// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_GTEST_RESOURCES_HPP_
#define GKO_CORE_TEST_GTEST_RESOURCES_HPP_


#include <gtest/gtest.h>


class ResourceEnvironment : public ::testing::Environment {
public:
    explicit ResourceEnvironment(int rank = 0, int size = 1);

    static int omp_threads;
    static int cuda_device_id;
    static int hip_device_id;
    static int sycl_device_id;
};


#endif  // GKO_CORE_TEST_GTEST_RESOURCES_HPP_
