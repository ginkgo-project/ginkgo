/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <core/base/exception_helpers.hpp>


#include <gtest/gtest.h>


namespace {


void not_implemented_func() NOT_IMPLEMENTED;


TEST(NotImplemented, ThrowsWhenUsed)
{
    ASSERT_THROW(not_implemented_func(), gko::NotImplemented);
}


void not_compiled_func() NOT_COMPILED(omp);


TEST(NotCompiled, ThrowsWhenUsed)
{
    ASSERT_THROW(not_compiled_func(), gko::NotCompiled);
}


void does_not_support_int() { throw NOT_SUPPORTED(int); }


TEST(NotSupported, ReturnsNotSupportedException)
{
    ASSERT_THROW(does_not_support_int(), gko::NotSupported);
}


void throws_cuda_error() { throw CUDA_ERROR(0); }

TEST(CudaError, ReturnsCudaError)
{
    ASSERT_THROW(throws_cuda_error(), gko::CudaError);
}


void throws_cublas_error() { throw CUBLAS_ERROR(0); }

TEST(CudaError, ReturnsCublasError)
{
    ASSERT_THROW(throws_cublas_error(), gko::CublasError);
}


void throws_cusparse_error() { throw CUSPARSE_ERROR(0); }

TEST(CudaError, ReturnsCusparseError)
{
    ASSERT_THROW(throws_cusparse_error(), gko::CusparseError);
}


TEST(AssertConformant, DoesNotThrowWhenConformant)
{
    ASSERT_NO_THROW(ASSERT_CONFORMANT(gko::dim<2>(3, 5), gko::dim<2>(5, 6)));
}


TEST(AssertConformant, ThrowsWhenNotConformant)
{
    ASSERT_THROW(ASSERT_CONFORMANT(gko::dim<2>(3, 5), gko::dim<2>(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualRows, DoesNotThrowWhenEqualRowSize)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_ROWS(gko::dim<2>(5, 3), gko::dim<2>(5, 6)));
}


TEST(AssertEqualRows, ThrowsWhenDifferentRowSize)
{
    ASSERT_THROW(ASSERT_EQUAL_ROWS(gko::dim<2>(3, 5), gko::dim<2>(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualCols, DoesNotThrowWhenEqualColSize)
{
    ASSERT_NO_THROW(ASSERT_EQUAL_COLS(gko::dim<2>(3, 6), gko::dim<2>(5, 6)));
}


TEST(AssertEqualCols, ThrowsWhenDifferentColSize)
{
    ASSERT_THROW(ASSERT_EQUAL_COLS(gko::dim<2>(3, 5), gko::dim<2>(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualDimensions, DoesNotThrowWhenEqualDimensions)
{
    ASSERT_NO_THROW(
        ASSERT_EQUAL_DIMENSIONS(gko::dim<2>(5, 6), gko::dim<2>(5, 6)));
}


TEST(AssertEqualDimensions, ThrowsWhenDifferentDimensions)
{
    ASSERT_THROW(ASSERT_EQUAL_DIMENSIONS(gko::dim<2>(3, 5), gko::dim<2>(7, 5)),
                 gko::DimensionMismatch);
}


TEST(EnsureAllocated, DoesNotThrowWhenAllocated)
{
    int x = 5;
    ASSERT_NO_THROW(ENSURE_ALLOCATED(&x, "OMP", 4));
}


TEST(EnsureAllocated, ThrowsWhenNotAllocated)
{
    ASSERT_THROW(ENSURE_ALLOCATED(nullptr, "OMP", 20), gko::AllocationError);
}


TEST(EnsureInBounds, DoesNotThrowWhenInBounds)
{
    ASSERT_NO_THROW(ENSURE_IN_BOUNDS(9, 10));
}

TEST(EnsureInBounds, ThrowWhenOutOfBounds)
{
    ASSERT_THROW(ENSURE_IN_BOUNDS(10, 10), gko::OutOfBoundsError);
}

void func_with_stream_error() { throw STREAM_ERROR("error message"); }

TEST(StreamError, ThrowsStreamErrorException)
{
    ASSERT_THROW(func_with_stream_error(), gko::StreamError);
}


void non_existing_kernel() KERNEL_NOT_FOUND;

TEST(KernelNotFound, ThrowsKernelNotFoundException)
{
    ASSERT_THROW(non_existing_kernel(), gko::KernelNotFound);
}


}  // namespace
