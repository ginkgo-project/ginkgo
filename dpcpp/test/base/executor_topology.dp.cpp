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


#include <exception>
#include <memory>
#include <type_traits>


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace {


class DpcppExecutor : public ::testing::Test {
protected:
    DpcppExecutor()
        : omp(gko::OmpExecutor::create()), dpcpp(nullptr), dpcpp2(nullptr)
    {}

    void SetUp()
    {
        ASSERT_GT(gko::DpcppExecutor::get_num_devices("cpu"), 0);
        dpcpp = gko::DpcppExecutor::create(0, omp, "cpu");
        if (gko::DpcppExecutor::get_num_devices("gpu") > 0) {
            dpcpp2 = gko::DpcppExecutor::create(0, omp, "gpu");
        }
    }

    void TearDown()
    {
        if (dpcpp != nullptr) {
            // ensure that previous calls finished and didn't throw an error
            ASSERT_NO_THROW(dpcpp->synchronize());
        }
    }

    std::shared_ptr<gko::Executor> omp;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp;
    std::shared_ptr<const gko::DpcppExecutor> dpcpp2;
};


TEST_F(DpcppExecutor, CanGetExecInfo)
{
    dpcpp = gko::DpcppExecutor::create(0, omp);

    auto exec_info = dpcpp->get_exec_info();

    ASSERT_TRUE(exec_info.max_workitem_sizes.size() > 0);
}


}  // namespace
