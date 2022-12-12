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
