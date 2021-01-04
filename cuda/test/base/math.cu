/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include <ginkgo/core/base/math.hpp>


#include <cmath>
#include <complex>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/test/utils.hpp"


namespace {
namespace kernel {


template <typename T, typename FuncType>
__device__ bool test_real_is_finite_function(FuncType isfin)
{
    constexpr T inf = gko::device_numeric_limits<T>::inf;
    constexpr T quiet_nan = NAN;
    bool test_true{};
    bool test_false{};

    test_true = isfin(T{0}) && isfin(-T{0}) && isfin(T{1});
    test_false = isfin(inf) || isfin(-inf) || isfin(quiet_nan) ||
                 isfin(inf - inf) || isfin(inf / inf) || isfin(inf * T{2}) ||
                 isfin(T{1} / T{0}) || isfin(T{0} / T{0});
    return test_true && !test_false;
}


template <typename ComplexType, typename FuncType>
__device__ bool test_complex_is_finite_function(FuncType isfin)
{
    static_assert(gko::is_complex_s<ComplexType>::value,
                  "Template type must be a complex type.");
    using T = gko::remove_complex<ComplexType>;
    using c_type = gko::kernels::cuda::cuda_type<ComplexType>;
    constexpr T inf = gko::device_numeric_limits<T>::inf;
    constexpr T quiet_nan = NAN;
    bool test_true{};
    bool test_false{};

    test_true = isfin(c_type{T{0}, T{0}}) && isfin(c_type{-T{0}, -T{0}}) &&
                isfin(c_type{T{1}, T{0}}) && isfin(c_type{T{0}, T{1}});
    test_false = isfin(c_type{inf, T{0}}) || isfin(c_type{-inf, T{0}}) ||
                 isfin(c_type{quiet_nan, T{0}}) || isfin(c_type{T{0}, inf}) ||
                 isfin(c_type{T{0}, -inf}) || isfin(c_type{T{0}, quiet_nan});
    return test_true && !test_false;
}


}  // namespace kernel


template <typename T>
__global__ void test_real_is_finite(bool *result)
{
    *result = kernel::test_real_is_finite_function<T>(
        [](T val) { return gko::is_finite(val); });
}


template <typename ComplexType>
__global__ void test_complex_is_finite(bool *result)
{
    *result = kernel::test_complex_is_finite_function<ComplexType>(
        [](ComplexType val) { return gko::is_finite(val); });
}


class IsFinite : public ::testing::Test {
protected:
    IsFinite()
        : ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref))
    {}

    template <typename T>
    bool test_real_is_finite_kernel()
    {
        gko::Array<bool> result(cuda, 1);
        test_real_is_finite<T><<<1, 1>>>(result.get_data());
        result.set_executor(ref);
        return *result.get_data();
    }

    template <typename T>
    bool test_complex_is_finite_kernel()
    {
        gko::Array<bool> result(cuda, 1);
        test_complex_is_finite<T><<<1, 1>>>(result.get_data());
        result.set_executor(ref);
        return *result.get_data();
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
};


TEST_F(IsFinite, Float) { ASSERT_TRUE(test_real_is_finite_kernel<float>()); }


TEST_F(IsFinite, Double) { ASSERT_TRUE(test_real_is_finite_kernel<double>()); }


TEST_F(IsFinite, FloatComplex)
{
    ASSERT_TRUE(test_complex_is_finite_kernel<thrust::complex<float>>());
}


TEST_F(IsFinite, DoubleComplex)
{
    ASSERT_TRUE(test_complex_is_finite_kernel<thrust::complex<double>>());
}


}  // namespace
