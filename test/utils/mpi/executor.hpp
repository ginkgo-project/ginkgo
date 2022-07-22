/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::ReferenceExecutor>& exec)
{
    exec = gko::ReferenceExecutor::create();
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::OmpExecutor>& exec)
{
    exec = gko::OmpExecutor::create();
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::CudaExecutor>& exec)
{
    ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
    exec = gko::CudaExecutor::create(
        gko::mpi::map_rank_to_device_id(MPI_COMM_WORLD,
                                        gko::CudaExecutor::get_num_devices()),
        ref);
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::HipExecutor>& exec)
{
    ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
    exec = gko::HipExecutor::create(
        gko::mpi::map_rank_to_device_id(MPI_COMM_WORLD,
                                        gko::HipExecutor::get_num_devices()),
        ref);
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::DpcppExecutor>& exec)
{
    auto num_gpu_devices = gko::DpcppExecutor::get_num_devices("gpu");
    auto num_cpu_devices = gko::DpcppExecutor::get_num_devices("cpu");
    if (num_gpu_devices > 0) {
        exec = gko::DpcppExecutor::create(
            gko::mpi::map_rank_to_device_id(MPI_COMM_WORLD, num_gpu_devices),
            ref, "gpu");
    } else if (num_cpu_devices > 0) {
        exec = gko::DpcppExecutor::create(
            gko::mpi::map_rank_to_device_id(MPI_COMM_WORLD, num_cpu_devices),
            ref, "cpu");
    } else {
        FAIL() << "No suitable DPC++ devices";
    }
}


#endif  // GKO_TEST_UTILS_MPI_EXECUTOR_HPP_
