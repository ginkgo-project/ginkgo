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

#ifndef GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
#define GKO_TEST_UTILS_MPI_EXECUTOR_HPP_


#include <ginkgo/core/base/executor.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/mpi.hpp>


template <typename ExecType>
std::shared_ptr<ExecType> init_executor(
    std::shared_ptr<gko::ReferenceExecutor>);


template <>
inline std::shared_ptr<gko::ReferenceExecutor>
init_executor<gko::ReferenceExecutor>(std::shared_ptr<gko::ReferenceExecutor>)
{
    return gko::ReferenceExecutor::create();
}


template <>
inline std::shared_ptr<gko::OmpExecutor> init_executor<gko::OmpExecutor>(
    std::shared_ptr<gko::ReferenceExecutor>)
{
    return gko::OmpExecutor::create();
}


template <>
inline std::shared_ptr<gko::CudaExecutor> init_executor<gko::CudaExecutor>(
    std::shared_ptr<gko::ReferenceExecutor> ref)
{
    {
        if (gko::CudaExecutor::get_num_devices() == 0) {
            throw std::runtime_error{"No suitable CUDA devices"};
        }
        return gko::CudaExecutor::create(
            gko::experimental::mpi::map_rank_to_device_id(
                MPI_COMM_WORLD, gko::CudaExecutor::get_num_devices()),
            ref);
    }
}


template <>
inline std::shared_ptr<gko::HipExecutor> init_executor<gko::HipExecutor>(
    std::shared_ptr<gko::ReferenceExecutor> ref)
{
    if (gko::HipExecutor::get_num_devices() == 0) {
        throw std::runtime_error{"No suitable HIP devices"};
    }
    return gko::HipExecutor::create(
        gko::experimental::mpi::map_rank_to_device_id(
            MPI_COMM_WORLD, gko::HipExecutor::get_num_devices()),
        ref);
}


template <>
inline std::shared_ptr<gko::DpcppExecutor> init_executor<gko::DpcppExecutor>(
    std::shared_ptr<gko::ReferenceExecutor> ref)
{
    auto num_gpu_devices = gko::DpcppExecutor::get_num_devices("gpu");
    auto num_cpu_devices = gko::DpcppExecutor::get_num_devices("cpu");
    if (num_gpu_devices > 0) {
        return gko::DpcppExecutor::create(
            gko::experimental::mpi::map_rank_to_device_id(MPI_COMM_WORLD,
                                                          num_gpu_devices),
            ref, "gpu");
    } else if (num_cpu_devices > 0) {
        return gko::DpcppExecutor::create(
            gko::experimental::mpi::map_rank_to_device_id(MPI_COMM_WORLD,
                                                          num_cpu_devices),
            ref, "cpu");
    } else {
        throw std::runtime_error{"No suitable DPC++ devices"};
    }
}


class CommonMpiTestFixture : public ::testing::Test {
public:
#if GINKGO_COMMON_SINGLE_MODE
    using value_type = float;
#else
    using value_type = double;
#endif
    using index_type = int;

    CommonMpiTestFixture()
        : ref{gko::ReferenceExecutor::create()},
          exec{init_executor<gko::EXEC_TYPE>(ref)},
          comm(MPI_COMM_WORLD)
    {}

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;

    gko::experimental::mpi::communicator comm;
};


#endif  // GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
