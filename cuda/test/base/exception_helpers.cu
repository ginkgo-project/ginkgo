// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/exception_helpers.hpp>


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <cusparse.h>


#include <gtest/gtest.h>


namespace {


TEST(AssertNoCudaErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUDA_ERRORS(1), gko::CudaError);
}


TEST(AssertNoCudaErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUDA_ERRORS(cudaSuccess));
}


TEST(AssertNoCublasErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUBLAS_ERRORS(1), gko::CublasError);
}


TEST(AssertNoCublasErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUBLAS_ERRORS(CUBLAS_STATUS_SUCCESS));
}


TEST(AssertNoCurandErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CURAND_ERRORS(1), gko::CurandError);
}


TEST(AssertNoCurandErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CURAND_ERRORS(CURAND_STATUS_SUCCESS));
}


TEST(AssertNoCusparseErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUSPARSE_ERRORS(1), gko::CusparseError);
}


TEST(AssertNoCusparseErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUSPARSE_ERRORS(CUSPARSE_STATUS_SUCCESS));
}


TEST(AssertNoCufftErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUFFT_ERRORS(1), gko::CufftError);
}


TEST(AssertNoCufftErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUFFT_ERRORS(CUFFT_SUCCESS));
}


}  // namespace
