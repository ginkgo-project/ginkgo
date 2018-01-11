#include <core/base/exception.hpp>


#include <gtest/gtest.h>


namespace {


TEST(ExceptionClasses, ErrorReturnsCorrectWhatMessage)
{
    gko::Error error("test_file.cpp", 1, "test error");
    ASSERT_EQ(std::string("test_file.cpp:1: test error"), error.what());
}


TEST(ExceptionClasses, NotImplementedReturnsCorrectWhatMessage)
{
    gko::NotImplemented error("test_file.cpp", 25, "test_func");
    ASSERT_EQ(std::string("test_file.cpp:25: test_func is not implemented"),
              error.what());
}


TEST(ExceptionClasses, NotCompiledReturnsCorrectWhatMessage)
{
    gko::NotCompiled error("test_file.cpp", 345, "test_func", "nvidia");
    ASSERT_EQ(std::string("test_file.cpp:345: feature test_func is part of the "
                          "nvidia module, which is not compiled on this "
                          "system"),
              error.what());
}


TEST(ExceptionClasses, NotSupportedReturnsCorrectWhatMessage)
{
    gko::NotSupported error("test_file.cpp", 123, "test_func", "test_obj");
    ASSERT_EQ(
        std::string("test_file.cpp:123: Operation test_func does not support "
                    "parameters of type test_obj"),
        error.what());
}


TEST(ExceptionClasses, CudaErrorReturnsCorrectWhatMessage)
{
    gko::CudaError error("test_file.cpp", 123, "test_func", 1);
    std::string expected = "test_file.cpp:123: test_func: ";
    ASSERT_EQ(expected, std::string(error.what()).substr(0, expected.size()));
}


TEST(ExceptionClasses, DimensionMismatchReturnsCorrectWhatMessage)
{
    gko::DimensionMismatch error("test_file.cpp", 243, "test_func", 3, 4, 2, 5);
    ASSERT_EQ(std::string("test_file.cpp:243: test_func: attempting to apply a "
                          "[3 x 4] operator on a [2 x 5] batch of vectors"),
              error.what());
}


TEST(ExceptionClasses, NotFoundReturnsCorrectWhatMessage)
{
    gko::NotFound error("test_file.cpp", 195, "my_func", "my error");
    ASSERT_EQ(std::string("test_file.cpp:195: my_func: my error"),
              error.what());
}


TEST(ExceptionClasses, AllocationErrorReturnsCorrectWhatMessage)
{
    gko::AllocationError error("test_file.cpp", 42, "CPU", 135);
    ASSERT_EQ(
        std::string("test_file.cpp:42: CPU: failed to allocate memory block "
                    "of 135B"),
        error.what());
}


}  // namespace
