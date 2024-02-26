// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
#define GKO_TEST_UTILS_MPI_EXECUTOR_HPP_


#include <ginkgo/core/base/executor.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/mpi.hpp>


#include "test/utils/executor.hpp"


class CommonMpiTestFixture : public ::testing::Test {
public:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = int;

    CommonMpiTestFixture()
        : comm(MPI_COMM_WORLD),
#ifdef GKO_COMPILING_CUDA
          stream(ResourceEnvironment::cuda_device_id),
#endif
#ifdef GKO_COMPILING_HIP
          stream(ResourceEnvironment::hip_device_id),
#endif
          ref{gko::ReferenceExecutor::create()}
    {
#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
        init_executor(ref, exec, stream.get());
#else
        init_executor(ref, exec);
#endif
        guard = exec->get_scoped_device_id_guard();
    }

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    gko::experimental::mpi::communicator comm;

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


#endif  // GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
