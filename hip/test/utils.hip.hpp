// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_TEST_UTILS_HIP_HPP_
#define GKO_HIP_TEST_UTILS_HIP_HPP_


#include "core/test/utils.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>


#include "hip/base/device.hpp"


namespace {


class HipEnvironment : public ::testing::Environment {
public:
    void TearDown() override { gko::kernels::hip::reset_device(0); }
};

testing::Environment* hip_env =
    testing::AddGlobalTestEnvironment(new HipEnvironment);


class HipTestFixture : public ::testing::Test {
protected:
    HipTestFixture()
        : ref(gko::ReferenceExecutor::create()),
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(0),
          exec(gko::HipExecutor::create(
              0, ref, std::make_shared<gko::HipAllocator>(), stream.get()))
#else
          exec(gko::HipExecutor::create(0, ref))
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
    gko::hip_stream stream;
#endif
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> exec;
};


}  // namespace


#endif  // GKO_HIP_TEST_UTILS_HIP_HPP_
