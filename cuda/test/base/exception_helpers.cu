/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>


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


TEST(AssertNoCusparseErrors, ThrowsOnError)
{
    ASSERT_THROW(GKO_ASSERT_NO_CUSPARSE_ERRORS(1), gko::CusparseError);
}


TEST(AssertNoCusparseErrors, DoesNotThrowOnSuccess)
{
    ASSERT_NO_THROW(GKO_ASSERT_NO_CUSPARSE_ERRORS(CUSPARSE_STATUS_SUCCESS));
}


}  // namespace
