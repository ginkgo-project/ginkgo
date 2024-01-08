// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_GTEST_ENVIRONMENTS_HPP_
#define GKO_CORE_TEST_GTEST_ENVIRONMENTS_HPP_


#include <algorithm>
#include <regex>
#include <sstream>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/mpi.hpp>


#include "core/test/gtest/resources.hpp"


#ifdef GKO_COMPILING_OMP
#include <omp.h>
#endif


#ifdef GKO_COMPILING_CUDA
#include "cuda/base/device.hpp"
#endif


#ifdef GKO_COMPILING_HIP
#include "hip/base/device.hpp"
#endif


#if GKO_COMPILING_DPCPP
#include "dpcpp/base/device.hpp"
#endif


class DeviceEnvironment : public ::testing::Environment {
public:
    explicit DeviceEnvironment(int rank) : rank_(rank) { print_environment(); }

#ifdef GKO_COMPILING_OMP
    void print_environment() const
    {
        if (ResourceEnvironment::omp_threads > 0) {
            omp_set_num_threads(ResourceEnvironment::omp_threads);
        }
        std::stringstream ss;
        ss << "Rank " << rank_ << ": OMP threads " << omp_get_max_threads()
           << std::endl;
        std::cerr << ss.str();
    }
#elif defined(GKO_COMPILING_CUDA)
    void print_environment() const
    {
        auto device_id = ResourceEnvironment::cuda_device_id;
        std::stringstream ss;
        ss << "Rank " << rank_ << ": CUDA device "
           << gko::kernels::cuda::get_device_name(device_id) << " ID "
           << device_id << std::endl;
        std::cerr << ss.str();
    }

    void TearDown() override
    {
        gko::kernels::cuda::reset_device(ResourceEnvironment::cuda_device_id);
    }
#elif defined(GKO_COMPILING_HIP)
    void print_environment() const
    {
        auto device_id = ResourceEnvironment::hip_device_id;
        std::stringstream ss;
        ss << "Rank " << rank_ << ": HIP device "
           << gko::kernels::hip::get_device_name(device_id) << " ID "
           << device_id << std::endl;
        std::cerr << ss.str();
    }

    void TearDown() override
    {
        gko::kernels::hip::reset_device(ResourceEnvironment::hip_device_id);
    }
#elif defined(GKO_COMPILING_DPCPP)
    void print_environment() const
    {
        auto device_id = ResourceEnvironment::sycl_device_id;
        std::stringstream ss;
        ss << "Rank " << rank_ << ": SYCL device "
           << gko::kernels::dpcpp::get_device_name(device_id) << " ID "
           << device_id << std::endl;
        std::cerr << ss.str();
    }
#else
    void print_environment() const {}
#endif

private:
    int rank_;
};


#endif  // GKO_CORE_TEST_GTEST_ENVIRONMENTS_HPP_
