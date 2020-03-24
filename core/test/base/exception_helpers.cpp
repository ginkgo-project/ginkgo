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


namespace {


void not_implemented_func() GKO_NOT_IMPLEMENTED;

TEST(NotImplemented, ThrowsWhenUsed)
{
    ASSERT_THROW(not_implemented_func(), gko::NotImplemented);
}


void not_compiled_func() GKO_NOT_COMPILED(omp);

TEST(NotCompiled, ThrowsWhenUsed)
{
    ASSERT_THROW(not_compiled_func(), gko::NotCompiled);
}


template <typename Expected, typename T>
void test_not_supported_impl(const T &obj)
{
    try {
        GKO_NOT_SUPPORTED(obj);
        FAIL();
    } catch (gko::NotSupported &m) {
        // check for equal suffix
        std::string msg{m.what()};
        auto expected = gko::name_demangling::get_type_name(typeid(Expected));
        ASSERT_TRUE(
            std::equal(expected.rbegin(), expected.rend(), msg.rbegin()));
    }
}


TEST(NotSupported, ReturnsIntNotSupportedException)
{
    test_not_supported_impl<int>(int{});
}


struct Base {
    virtual ~Base() = default;
};

struct Derived : Base {};


TEST(NotSupported, ReturnsPtrNotSupportedException)
{
    Derived d;
    Base *b = &d;
    test_not_supported_impl<Derived>(b);
}


TEST(NotSupported, ReturnsRefNotSupportedException)
{
    Derived d;
    Base &b = d;
    test_not_supported_impl<Derived>(b);
}


void throws_cuda_error() { throw GKO_CUDA_ERROR(0); }

TEST(CudaError, ReturnsCudaError)
{
    ASSERT_THROW(throws_cuda_error(), gko::CudaError);
}


void throws_cublas_error() { throw GKO_CUBLAS_ERROR(0); }

TEST(CudaError, ReturnsCublasError)
{
    ASSERT_THROW(throws_cublas_error(), gko::CublasError);
}


void throws_cusparse_error() { throw GKO_CUSPARSE_ERROR(0); }

TEST(CudaError, ReturnsCusparseError)
{
    ASSERT_THROW(throws_cusparse_error(), gko::CusparseError);
}


TEST(AssertIsSquareMatrix, DoesNotThrowWhenIsSquareMatrix)
{
    ASSERT_NO_THROW(GKO_ASSERT_IS_SQUARE_MATRIX(gko::dim<2>(3, 3)));
}


TEST(AssertIsSquareMatrix, ThrowsWhenIsNotSquareMatrix)
{
    ASSERT_THROW(GKO_ASSERT_IS_SQUARE_MATRIX(gko::dim<2>(3, 4)),
                 gko::DimensionMismatch);
}


TEST(AssertIsNonEmptymatrix, DoesNotThrowWhenIsNonEmptyMatrix)
{
    ASSERT_NO_THROW(GKO_ASSERT_IS_NON_EMPTY_MATRIX(gko::dim<2>(1, 1)));
}


TEST(AssertIsNonEmptyMatrix, ThrowsWhenIsEmptyMatrix)
{
    ASSERT_THROW(GKO_ASSERT_IS_NON_EMPTY_MATRIX(gko::dim<2>(0, 0)),
                 gko::BadDimension);
    ASSERT_THROW(GKO_ASSERT_IS_NON_EMPTY_MATRIX(gko::dim<2>(1, 0)),
                 gko::BadDimension);
    ASSERT_THROW(GKO_ASSERT_IS_NON_EMPTY_MATRIX(gko::dim<2>(0, 1)),
                 gko::BadDimension);
}


TEST(AssertConformant, DoesNotThrowWhenConformant)
{
    ASSERT_NO_THROW(
        GKO_ASSERT_CONFORMANT(gko::dim<2>(3, 5), gko::dim<2>(5, 6)));
}


TEST(AssertConformant, ThrowsWhenNotConformant)
{
    ASSERT_THROW(GKO_ASSERT_CONFORMANT(gko::dim<2>(3, 5), gko::dim<2>(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqual, DoesNotThrowWhenEqual)
{
    ASSERT_NO_THROW(GKO_ASSERT_EQ(1, 1));
}


TEST(AssertEqual, ThrowsWhenNotEqual)
{
    ASSERT_THROW(GKO_ASSERT_EQ(0, 1), gko::ValueMismatch);
}


TEST(AssertEqualRows, DoesNotThrowWhenEqualRowSize)
{
    ASSERT_NO_THROW(
        GKO_ASSERT_EQUAL_ROWS(gko::dim<2>(5, 3), gko::dim<2>(5, 6)));
}


TEST(AssertEqualRows, ThrowsWhenDifferentRowSize)
{
    ASSERT_THROW(GKO_ASSERT_EQUAL_ROWS(gko::dim<2>(3, 5), gko::dim<2>(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualCols, DoesNotThrowWhenEqualColSize)
{
    ASSERT_NO_THROW(
        GKO_ASSERT_EQUAL_COLS(gko::dim<2>(3, 6), gko::dim<2>(5, 6)));
}


TEST(AssertEqualCols, ThrowsWhenDifferentColSize)
{
    ASSERT_THROW(GKO_ASSERT_EQUAL_COLS(gko::dim<2>(3, 5), gko::dim<2>(7, 3)),
                 gko::DimensionMismatch);
}


TEST(AssertEqualDimensions, DoesNotThrowWhenEqualDimensions)
{
    ASSERT_NO_THROW(
        GKO_ASSERT_EQUAL_DIMENSIONS(gko::dim<2>(5, 6), gko::dim<2>(5, 6)));
}


TEST(AssertEqualDimensions, ThrowsWhenDifferentDimensions)
{
    ASSERT_THROW(
        GKO_ASSERT_EQUAL_DIMENSIONS(gko::dim<2>(3, 5), gko::dim<2>(7, 5)),
        gko::DimensionMismatch);
}


TEST(EnsureAllocated, DoesNotThrowWhenAllocated)
{
    int x = 5;
    ASSERT_NO_THROW(GKO_ENSURE_ALLOCATED(&x, "OMP", 4));
}


TEST(EnsureAllocated, ThrowsWhenNotAllocated)
{
    ASSERT_THROW(GKO_ENSURE_ALLOCATED(nullptr, "OMP", 20),
                 gko::AllocationError);
}


TEST(EnsureInBounds, DoesNotThrowWhenInBounds)
{
    ASSERT_NO_THROW(GKO_ENSURE_IN_BOUNDS(9, 10));
}


TEST(EnsureInBounds, ThrowWhenOutOfBounds)
{
    ASSERT_THROW(GKO_ENSURE_IN_BOUNDS(10, 10), gko::OutOfBoundsError);
}


void func_with_stream_error() { throw GKO_STREAM_ERROR("error message"); }

TEST(StreamError, ThrowsStreamErrorException)
{
    ASSERT_THROW(func_with_stream_error(), gko::StreamError);
}


void non_existing_kernel() GKO_KERNEL_NOT_FOUND;

TEST(KernelNotFound, ThrowsKernelNotFoundException)
{
    ASSERT_THROW(non_existing_kernel(), gko::KernelNotFound);
}


}  // namespace
