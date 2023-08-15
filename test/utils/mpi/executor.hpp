// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
#define GKO_TEST_UTILS_MPI_EXECUTOR_HPP_


#include <ginkgo/core/base/executor.hpp>


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/mpi.hpp>
#include <ginkgo/core/base/stream.hpp>


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
            gko::experimental::mpi::map_rank_to_device_id(
                MPI_COMM_WORLD, gko::CudaExecutor::get_num_devices()),
            ref, std::make_shared<gko::CudaAllocator>(), stream);
    }
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::HipExecutor>& exec,
                          GKO_HIP_STREAM_STRUCT* stream = nullptr)
{
    if (gko::HipExecutor::get_num_devices() == 0) {
        throw std::runtime_error{"No suitable HIP devices"};
    }
    exec = gko::HipExecutor::create(
        gko::experimental::mpi::map_rank_to_device_id(
            MPI_COMM_WORLD, gko::HipExecutor::get_num_devices()),
        ref, std::make_shared<gko::HipAllocator>(), stream);
}


inline void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                          std::shared_ptr<gko::DpcppExecutor>& exec)
{
    auto num_gpu_devices = gko::DpcppExecutor::get_num_devices("gpu");
    auto num_cpu_devices = gko::DpcppExecutor::get_num_devices("cpu");
    if (num_gpu_devices > 0) {
        exec = gko::DpcppExecutor::create(
            gko::experimental::mpi::map_rank_to_device_id(MPI_COMM_WORLD,
                                                          num_gpu_devices),
            ref, "gpu");
    } else if (num_cpu_devices > 0) {
        exec = gko::DpcppExecutor::create(
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
        : comm(MPI_COMM_WORLD),
#if defined(GKO_TEST_NONDEFAULT_STREAM) && \
    (defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP))

          stream(gko::experimental::mpi::map_rank_to_device_id(
              comm.get(), gko::EXEC_TYPE::get_num_devices())),
#endif
          ref{gko::ReferenceExecutor::create()}
    {
#if defined(GKO_TEST_NONDEFAULT_STREAM) && \
    (defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP))
        init_executor(ref, exec, stream.get());
#else
        init_executor(ref, exec);
#endif
    }

    void TearDown() final
    {
        if (exec != nullptr) {
            ASSERT_NO_THROW(exec->synchronize());
        }
    }

    gko::experimental::mpi::communicator comm;

#ifdef GKO_TEST_NONDEFAULT_STREAM
#ifdef GKO_COMPILING_CUDA
    gko::cuda_stream stream;
#endif
#ifdef GKO_COMPILING_HIP
    gko::hip_stream stream;
#endif
#endif

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::EXEC_TYPE> exec;
};


#endif  // GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
