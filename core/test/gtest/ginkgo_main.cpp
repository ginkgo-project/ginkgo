#include <gtest/gtest.h>


#include "core/test/gtest/environments.hpp"

resource ResourceEnvironment::rs = {};

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    auto resources = get_ctest_resources();

    ::testing::AddGlobalTestEnvironment(
        new ResourceEnvironment(resources.front()));
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    ::testing::AddGlobalTestEnvironment(new HipEnvironment);
    ::testing::AddGlobalTestEnvironment(new OmpEnvironment);
    int result = RUN_ALL_TESTS();
    return result;
}