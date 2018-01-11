#include <core/base/exception_helpers.hpp>


#include <gtest/gtest.h>


namespace {


TEST(AssertNoCudaErrors, ThrowsOnError)
{
    ASSERT_THROW(ASSERT_NO_CUDA_ERRORS(1), gko::CudaError);
}


TEST(AssertNoCudaErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(ASSERT_NO_CUDA_ERRORS(cudaSuccess));
}


}  // namespace
