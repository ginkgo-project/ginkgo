/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/executor.hpp>


#include <memory>
#include <thread>
#include <type_traits>


#if defined(__unix__) || defined(__APPLE__)
#include <numa.h>
#include <utmpx.h>
#endif


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/test/utils.hpp"


namespace {


class CudaExecutor : public ::testing::Test {
protected:
    CudaExecutor()
        : omp(gko::OmpExecutor::create()), cuda(nullptr), cuda2(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::CudaExecutor::get_num_devices(), 0);
        cuda = gko::CudaExecutor::create(0, omp);
        cuda2 = gko::CudaExecutor::create(
            gko::CudaExecutor::get_num_devices() - 1, omp);
    }

    void TearDown()
    {
        if (cuda != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(cuda->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> omp;
    std::shared_ptr<gko::CudaExecutor> cuda;
    std::shared_ptr<gko::CudaExecutor> cuda2;
};


#if GKO_HAVE_HWLOC


inline int get_cpu_os_id(int log_id)
{
    return gko::get_machine_topology()->get_pu(log_id)->os_id;
}


inline int get_core_os_id(int log_id)
{
    return gko::get_machine_topology()->get_core(log_id)->os_id;
}


TEST_F(CudaExecutor, CanBindToCpus)
{
    auto cuda = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    auto cpu_sys = sched_getcpu();

    const int bind_cpu[2] = {6, 3};
    cuda->bind_to_cpus(bind_cpu, 2);

    cpu_sys = sched_getcpu();
    ASSERT_TRUE(cpu_sys == get_cpu_os_id(3) || cpu_sys == get_cpu_os_id(6));
}


TEST_F(CudaExecutor, CanBindToCores)
{
    auto cuda = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    auto cpu_sys = sched_getcpu();

    const int bind_cores[2] = {6, 3};
    cuda->bind_to_cores(bind_cores, 2);

    cpu_sys = sched_getcpu();
    ASSERT_TRUE(cpu_sys == get_core_os_id(3) || cpu_sys == get_core_os_id(6));
}


TEST_F(CudaExecutor, KnowsItsClosestCpus)
{
    auto cuda0 = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    auto cpu_sys = sched_getcpu();
    auto close_cpus0 = cuda0->get_closest_cpus();

    ASSERT_NE(close_cpus0[0], -1);
}


TEST_F(CudaExecutor, KnowsItsNuma)
{
    auto cuda0 = gko::CudaExecutor::create(0, gko::OmpExecutor::create());
    auto exec_info0 = cuda0->get_exec_info();
    auto numa0 = exec_info0.numa_node;
    auto close_cpu0 = cuda0->get_closest_cpus()[0];

    auto numa_sys0 = numa_node_of_cpu(get_cpu_os_id(close_cpu0));

    ASSERT_TRUE(numa0 == numa_sys0);
}


#endif


}  // namespace
