// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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
#include "test/utils/executor.hpp"


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

    void print_environment() const
    {
        auto ref = gko::ReferenceExecutor::create();
#ifdef GKO_COMPILING_OMP
        if (ResourceEnvironment::omp_threads > 0) {
            omp_set_num_threads(ResourceEnvironment::omp_threads);
        }
        std::shared_ptr<gko::OmpExecutor> exec;
#elif defined(GKO_COMPILING_CUDA)
        std::shared_ptr<gko::CudaExecutor> exec;
#elif defined(GKO_COMPILING_HIP)
        std::shared_ptr<gko::HipExecutor> exec;
#elif defined(GKO_COMPILING_DPCPP)
        std::shared_ptr<gko::DpcppExecutor> exec;
#else
        std::shared_ptr<gko::ReferenceExecutor> exec;
#endif
        init_executor(ref, exec);
        std::cerr << "Rank " << rank_ << ": " << exec->get_description()
                  << std::endl;
    }

private:
    int rank_;
};


#endif  // GKO_CORE_TEST_GTEST_ENVIRONMENTS_HPP_
