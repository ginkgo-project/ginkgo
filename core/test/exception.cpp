#include "core/exception.hpp"


#include <gtest/gtest.h>


namespace {


TEST(ExceptionClasses, ErrorReturnsCorrectWhatMessage)
{
    msparse::Error error("test_file.cpp", 1, "test error");
    ASSERT_EQ(std::string("test_file.cpp:1: test error"), error.what());
}


TEST(ExceptionClasses, NotImplementedReturnsCorrectWhatMessage)
{
    msparse::NotImplemented error("test_file.cpp", 25, "test_func");
    ASSERT_EQ(std::string("test_file.cpp:25: test_func is not implemented"),
              error.what());
}


TEST(ExceptionClasses, NotSupportedReturnsCorrectWhatMessage)
{
    msparse::NotSupported error("test_file.cpp", 123, "test_func", "test_obj");
    ASSERT_EQ(
        std::string("test_file.cpp:123: Operation test_func does not support "
                    "parameters of type test_obj"),
        error.what());
}


TEST(ExceptionClasses, DimensionMismatchReturnsCorrectWhatMessage)
{
    msparse::DimensionMismatch error("test_file.cpp", 243, "test_func", 3, 4, 2,
                                     5);
    ASSERT_EQ(std::string("test_file.cpp:243: test_func: attempting to apply a "
                          "[3 x 4] operator on a [2 x 5] batch of vectors"),
              error.what());
}


TEST(ExceptionClasses, NotFoundReturnsCorrectWhatMessage)
{
    msparse::NotFound error("test_file.cpp", 195, "my_func", "my error");
    ASSERT_EQ(std::string("test_file.cpp:195: my_func: my error"),
              error.what());
}


TEST(ExceptionClasses, AllocationErrorReturnsCorrectWhatMessage)
{
    msparse::AllocationError error("test_file.cpp", 42, "CPU", 135);
    ASSERT_EQ(
        std::string("test_file.cpp:42: CPU: failed to allocate memory block "
                    "of 135B"),
        error.what());
}


}  // namespace
