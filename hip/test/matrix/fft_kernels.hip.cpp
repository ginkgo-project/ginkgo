// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/fft.hpp>


#include <hip/hip_runtime.h>
#if HIP_VERSION >= 50200000
#include <hipfft/hipfft.h>
#else
#include <hipfft.h>
#endif


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


// since hipFFT is optional, we test the exception behavior here
TEST(AssertNoHipfftErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_HIPFFT_ERRORS(1), gko::HipfftError);
}


TEST(AssertNoHipfftErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_HIPFFT_ERRORS(HIPFFT_SUCCESS));
}
