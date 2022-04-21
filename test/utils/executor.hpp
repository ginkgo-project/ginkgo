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

#ifndef GKO_TEST_UTILS_EXECUTOR_HPP_
#define GKO_TEST_UTILS_EXECUTOR_HPP_


#include <ginkgo/core/base/executor.hpp>


#if GINKGO_BUILD_MPI
#include <ginkgo/core/base/mpi.hpp>
#endif


#include <memory>


#include <gtest/gtest.h>


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
    exec = gko::CudaExecutor::create(0, ref);
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::HipExecutor>& exec)
{
    ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
    exec = gko::HipExecutor::create(0, ref);
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::DpcppExecutor>& exec)
{
    if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
        exec = gko::DpcppExecutor::create(0, ref, "gpu");
    } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
        exec = gko::DpcppExecutor::create(0, ref, "cpu");
    } else {
        FAIL() << "No suitable DPC++ devices";
    }
}


#if GINKGO_BUILD_MPI


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::ReferenceExecutor>& exec,
                   gko::mpi::communicator comm)
{
    exec = gko::ReferenceExecutor::create();
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::OmpExecutor>& exec,
                   gko::mpi::communicator comm)
{
    exec = gko::OmpExecutor::create();
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::CudaExecutor>& exec,
                   gko::mpi::communicator comm)
{
    ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
    auto device_id =
        comm.node_local_rank() % gko::CudaExecutor::get_num_devices();
    exec = gko::CudaExecutor::create(device_id, ref);
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::HipExecutor>& exec,
                   gko::mpi::communicator comm)
{
    ASSERT_GT(gko::HipExecutor::get_num_devices(), 0);
    auto device_id =
        comm.node_local_rank() % gko::HipExecutor::get_num_devices();
    exec = gko::HipExecutor::create(device_id, ref);
}


void init_executor(std::shared_ptr<gko::ReferenceExecutor> ref,
                   std::shared_ptr<gko::DpcppExecutor>& exec,
                   gko::mpi::communicator comm)
{
    if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
        auto device_id =
            comm.node_local_rank() % gko::DpcppExecutor::get_num_devices("gpu");
        exec = gko::DpcppExecutor::create(device_id, ref);
    } else if (gko::DpcppExecutor::get_num_devices("cpu") > 0) {
        auto device_id =
            comm.node_local_rank() % gko::DpcppExecutor::get_num_devices("cpu");
        exec = gko::DpcppExecutor::create(device_id, ref);
    } else {
        FAIL() << "No suitable DPC++ devices";
    }
}


#endif


#endif  // GKO_TEST_UTILS_EXECUTOR_HPP_
