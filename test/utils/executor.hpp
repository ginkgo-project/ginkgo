// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TEST_UTILS_EXECUTOR_HPP_
#define GKO_TEST_UTILS_EXECUTOR_HPP_


#include <ginkgo/core/base/executor.hpp>


#include <memory>
#include <stdexcept>


#include <gtest/gtest.h>


#include <ginkgo/core/base/stream.hpp>


#include "core/test/gtest/resources.hpp"


#if GINKGO_COMMON_SINGLE_MODE
#define SKIP_IF_SINGLE_MODE GTEST_SKIP() << "Skip due to single mode"
#else
#define SKIP_IF_SINGLE_MODE                                                  \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor>,
                          std::shared_ptr<gko::ReferenceExecutor>& exec)
{
    exec = gko::ReferenceExecutor::create();
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor>,
                          std::shared_ptr<gko::OmpExecutor>& exec)
{
    exec = gko::OmpExecutor::create();
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::CudaExecutor>& exec,
                          CUstream_st* stream = nullptr)
{
    {
        if (gko::CudaExecutor::get_num_devices() == 0) {
            throw std::runtime_error{"No suitable CUDA devices"};
        }
        exec = gko::CudaExecutor::create(
            ResourceEnvironment::cuda_device_id, ref,
            std::make_shared<gko::CudaAllocator>(), stream);
    }
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::HipExecutor>& exec,
                          GKO_HIP_STREAM_STRUCT* stream = nullptr)
{
    if (gko::HipExecutor::get_num_devices() == 0) {
        throw std::runtime_error{"No suitable HIP devices"};
    }
    exec =
        gko::HipExecutor::create(ResourceEnvironment::hip_device_id, ref,
                                 std::make_shared<gko::HipAllocator>(), stream);
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::DpcppExecutor>& exec)
{
    if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
        exec = gko::DpcppExecutor::create(ResourceEnvironment::sycl_device_id,
                                          ref, "gpu");
    } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
        exec = gko::DpcppExecutor::create(0, ref, "cpu");
    } else {
        throw std::runtime_error{"No suitable DPC++ devices"};
    }
}


class CommonTestFixture : public ::testing::Test {
public:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = int;

    CommonTestFixture()
        :
#if defined(GKO_TEST_NONDEFAULT_STREAM) && defined(GKO_COMPILING_CUDA)
          stream(ResourceEnvironment::cuda_device_id),
#endif
#if defined(GKO_TEST_NONDEFAULT_STREAM) && defined(GKO_COMPILING_HIP)
          stream(ResourceEnvironment::hip_device_id),
#endif
          ref{gko::ReferenceExecutor::create()}
    {
#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
        init_executor(ref, exec, stream.get());
#else
        init_executor(ref, exec);
#endif
        // set device-id test-wide since some test call device
        // kernels directly
        guard = exec->get_scoped_device_id_guard();
    }

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

#ifdef GKO_COMPILING_CUDA
    gko::cuda_stream stream;
#endif
#ifdef GKO_COMPILING_HIP
    gko::hip_stream stream;
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
    gko::scoped_device_id_guard guard;
};


#endif  // GKO_TEST_UTILS_EXECUTOR_HPP_
