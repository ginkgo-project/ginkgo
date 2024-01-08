// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_TEST_UTILS_HIP_HPP_
#define GKO_HIP_TEST_UTILS_HIP_HPP_


#include "core/test/utils.hpp"


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>


#include "core/test/gtest/resources.hpp"
#include "hip/base/device.hpp"


namespace {


class HipTestFixture : public ::testing::Test {
protected:
    HipTestFixture()
        : ref(gko::ReferenceExecutor::create()),
#ifdef GKO_TEST_NONDEFAULT_STREAM
          stream(ResourceEnvironment::hip_device_id),
#endif
          exec(gko::HipExecutor::create(ResourceEnvironment::hip_device_id, ref,
                                        std::make_shared<gko::HipAllocator>(),
                                        stream.get())),
          guard(exec->get_scoped_device_id_guard())
    {}

    void TearDown()
    {
        if (exec != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            exec->synchronize();
        }
    }

    gko::hip_stream stream;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> exec;
    gko::scoped_device_id_guard guard;
};


}  // namespace


#endif  // GKO_HIP_TEST_UTILS_HIP_HPP_
