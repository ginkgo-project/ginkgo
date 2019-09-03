/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include <ginkgo/core/base/exception_helpers.hpp>


#include <gtest/gtest.h>


#include <hip/hip_runtime.h>
#include <hipblas.h>
#include <hipsparse.h>


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


TEST(AssertNoHipsparseErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_HIPSPARSE_ERRORS(1), gko::HipsparseError);
}


TEST(AssertNoHipsparseErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_HIPSPARSE_ERRORS(HIPSPARSE_STATUS_SUCCESS));
}


}  // namespace
