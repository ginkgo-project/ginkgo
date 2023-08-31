/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
