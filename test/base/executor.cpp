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

#include <ginkgo/core/base/executor.hpp>


#include <map>


#include <gtest/gtest.h>


#include "core/test/utils/assertions.hpp"
#include "test/utils/executor.hpp"


namespace reference {
int value = 5;
}

namespace omp {
int value = 1;
}

namespace cuda {
int value = 2;
}

namespace hip {
int value = 3;
}

namespace dpcpp {
int value = 4;
}


class ExampleOperation : public gko::Operation {
public:
    explicit ExampleOperation(int& val) : value(val) {}
    void run(std::shared_ptr<const gko::OmpExecutor>) const override
    {
        value = omp::value;
    }
    void run(std::shared_ptr<const gko::CudaExecutor>) const override
    {
        value = cuda::value;
    }
    void run(std::shared_ptr<const gko::HipExecutor>) const override
    {
        value = hip::value;
    }
    void run(std::shared_ptr<const gko::DpcppExecutor>) const override
    {
        value = dpcpp::value;
    }
    void run(std::shared_ptr<const gko::ReferenceExecutor>) const override
    {
        value = reference::value;
    }

    int& value;
};

class Executor : public CommonTestFixture {};


TEST_F(Executor, RunsCorrectOperation)
{
    int value = 0;

    exec->run(ExampleOperation(value));

    ASSERT_EQ(EXEC_NAMESPACE::value, value);
}


#ifndef GKO_COMPILING_REFERENCE


TEST_F(Executor, RunsCorrectLambdaOperation)
{
    int value = 0;
    auto omp_lambda = [&value]() { value = omp::value; };
    auto cuda_lambda = [&value]() { value = cuda::value; };
    auto hip_lambda = [&value]() { value = hip::value; };
    auto dpcpp_lambda = [&value]() { value = dpcpp::value; };

    exec->run(omp_lambda, cuda_lambda, hip_lambda, dpcpp_lambda);

    ASSERT_EQ(EXEC_NAMESPACE::value, value);
}


#endif  // GKO_COMPILING_REFERENCE
