#include <gtest/gtest.h>


#include "core/test/gtest/environments.hpp"


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnvironment);
    ::testing::AddGlobalTestEnvironment(new HipEnvironment);
    int result = RUN_ALL_TESTS();
    return result;
}