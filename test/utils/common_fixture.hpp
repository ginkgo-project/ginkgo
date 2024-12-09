// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TEST_UTILS_COMMON_FIXTURE_HPP_
#define GKO_TEST_UTILS_COMMON_FIXTURE_HPP_


#include <memory>
#include <stdexcept>

#include <gtest/gtest.h>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/stream.hpp>

#include "core/test/gtest/resources.hpp"
#include "test/utils/executor.hpp"


#if GINKGO_COMMON_SINGLE_MODE
#define SKIP_IF_SINGLE_MODE GTEST_SKIP() << "Skip due to single mode"
#else
#define SKIP_IF_SINGLE_MODE                                                  \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
#endif


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


#endif  // GKO_TEST_UTILS_COMMON_FIXTURE_HPP_
