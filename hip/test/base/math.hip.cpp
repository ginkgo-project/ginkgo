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

#include <ginkgo/core/base/math.hpp>


#include <cmath>
#include <complex>
#include <memory>


#include <gtest/gtest.h>
#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace {


template <typename T>
__global__ void test_real_isfinite(bool *result)
{
    constexpr T inf = INFINITY;
    bool test_true{};
    bool test_false{};

    test_true =
        gko::isfinite(T{0}) && gko::isfinite(-T{0}) && gko::isfinite(T{1});
    test_false = gko::isfinite(inf) || gko::isfinite(-inf) ||
                 gko::isfinite(NAN) || gko::isfinite(inf - inf) ||
                 gko::isfinite(inf / inf) || gko::isfinite(inf * T{2}) ||
                 gko::isfinite(T{1} / T{0}) || gko::isfinite(T{0} / T{0});
    *result = test_true && !test_false;
}


template <typename ComplexType>
__global__ void test_complex_isfinite(bool *result)
{
    static_assert(gko::is_complex_s<ComplexType>::value,
                  "Template type must be a complex type.");
    using T = gko::remove_complex<ComplexType>;
    using c_type = gko::kernels::hip::hip_type<ComplexType>;
    constexpr T inf = INFINITY;
    constexpr T quiet_nan = NAN;
    bool test_true{};
    bool test_false{};

    test_true = gko::isfinite(c_type{T{0}, T{0}}) &&
                gko::isfinite(c_type{-T{0}, -T{0}}) &&
                gko::isfinite(c_type{T{1}, T{0}}) &&
                gko::isfinite(c_type{T{0}, T{1}});
    test_false =
        gko::isfinite(c_type{inf, T{0}}) || gko::isfinite(c_type{-inf, T{0}}) ||
        gko::isfinite(c_type{quiet_nan, T{0}}) ||
        gko::isfinite(c_type{T{0}, inf}) || gko::isfinite(c_type{T{0}, -inf}) ||
        gko::isfinite(c_type{T{0}, quiet_nan});
    *result = test_true && !test_false;
}


class IsFinite : public ::testing::Test {
protected:
    IsFinite()
        : ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref))
    {}

    template <typename T>
    bool test_real_isfinite_kernel()
    {
        gko::Array<bool> result(hip, 1);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(test_real_isfinite<T>), dim3(1),
                           dim3(1), 0, 0, result.get_data());
        result.set_executor(ref);
        return *result.get_data();
    }

    template <typename T>
    bool test_complex_isfinite_kernel()
    {
        gko::Array<bool> result(hip, 1);
        hipLaunchKernelGGL(HIP_KERNEL_NAME(test_complex_isfinite<T>), dim3(1),
                           dim3(1), 0, 0, result.get_data());
        result.set_executor(ref);
        return *result.get_data();
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
};


TEST_F(IsFinite, Float) { ASSERT_TRUE(test_real_isfinite_kernel<float>()); }


TEST_F(IsFinite, Double) { ASSERT_TRUE(test_real_isfinite_kernel<double>()); }


TEST_F(IsFinite, FloatComplex)
{
    ASSERT_TRUE(test_complex_isfinite_kernel<thrust::complex<float>>());
}


TEST_F(IsFinite, DoubleComplex)
{
    ASSERT_TRUE(test_complex_isfinite_kernel<thrust::complex<double>>());
}


}  // namespace
