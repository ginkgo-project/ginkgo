// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TEST_UTILS_EXECUTOR_HPP_
#define GKO_TEST_UTILS_EXECUTOR_HPP_


#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>

#include "core/test/gtest/resources.hpp"


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


#endif  // GKO_TEST_UTILS_EXECUTOR_HPP_
