// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_TEST_UTILS_HPP_
#define GKO_CUDA_TEST_UTILS_HPP_


#include "core/test/utils.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>


#include "cuda/base/device.hpp"


namespace {


class CudaEnvironment : public ::testing::Environment {
public:
    void TearDown() override { gko::kernels::cuda::reset_device(0); }
};

testing::Environment* cuda_env =
    testing::AddGlobalTestEnvironment(new CudaEnvironment);


class CudaTestFixture : public ::testing::Test {
protected:
    CudaTestFixture()
        : ref(gko::ReferenceExecutor::create()),
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(0),
          exec(gko::CudaExecutor::create(
              0, ref, std::make_shared<gko::CudaAllocator>(), stream.get()))
#else
          exec(gko::CudaExecutor::create(0, ref))
#endif
    {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            exec->synchronize();
        }
    }

#ifdef GKO_TEST_NONDEFAULT_STREAM
    gko::cuda_stream stream;
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> exec;
};


}  // namespace


#endif  // GKO_CUDA_TEST_UTILS_HPP_
