// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>


#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <hipsparse/hipsparse.h>
#else
#include <hipblas.h>
#include <hiprand.h>
#include <hipsparse.h>
#endif


#include <gtest/gtest.h>


namespace {


TEST(AssertNoHipErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_HIP_ERRORS(1), gko::HipError);
}


TEST(AssertNoHipErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_HIP_ERRORS(hipSuccess));
}


TEST(AssertNoHipblasErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_HIPBLAS_ERRORS(1), gko::HipblasError);
}


TEST(AssertNoHipblasErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_HIPBLAS_ERRORS(HIPBLAS_STATUS_SUCCESS));
}


TEST(AssertNoHiprandErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_HIPRAND_ERRORS(1), gko::HiprandError);
}


TEST(AssertNoHiprandErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_HIPRAND_ERRORS(HIPRAND_STATUS_SUCCESS));
}


TEST(AssertNoHipsparseErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_HIPSPARSE_ERRORS(1), gko::HipsparseError);
}


TEST(AssertNoHipsparseErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_HIPSPARSE_ERRORS(HIPSPARSE_STATUS_SUCCESS));
}


}  // namespace
