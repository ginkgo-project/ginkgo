// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_TEST_UTILS_HPP_
#define GKO_CUDA_TEST_UTILS_HPP_


#include "core/test/utils.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>


#include "core/test/gtest/resources.hpp"
#include "cuda/base/device.hpp"


namespace {


class CudaTestFixture : public ::testing::Test {
protected:
    CudaTestFixture()
        : ref(gko::ReferenceExecutor::create()),
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(ResourceEnvironment::cuda_device_id),
#endif
          exec(gko::CudaExecutor::create(
              ResourceEnvironment::cuda_device_id, ref,
              std::make_shared<gko::CudaAllocator>(), stream.get())),
          guard(exec->get_scoped_device_id_guard())
    {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            exec->synchronize();
        }
    }

    gko::cuda_stream stream;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> exec;
    gko::scoped_device_id_guard guard;
};


}  // namespace


#endif  // GKO_CUDA_TEST_UTILS_HPP_
