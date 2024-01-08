// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
// prevent compilation failure related to disappearing assert(...) statements
#include <hip/hip_runtime.h>
// force-top: off


#include <ginkgo/core/base/math.hpp>


#include <cmath>
#include <complex>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/test/utils.hip.hpp"


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
    using c_type = gko::kernels::hip::hip_type<ComplexType>;
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
__global__ void test_real_is_finite(bool* result)
{
    *result = kernel::test_real_is_finite_function<T>(
        [](T val) { return gko::is_finite(val); });
}


template <typename ComplexType>
__global__ void test_complex_is_finite(bool* result)
{
    *result = kernel::test_complex_is_finite_function<ComplexType>(
        [](ComplexType val) { return gko::is_finite(val); });
}


class IsFinite : public HipTestFixture {
protected:
    template <typename T>
    bool test_real_is_finite_kernel()
    {
        gko::array<bool> result(exec, 1);
        test_real_is_finite<T>
            <<<1, 1, 0, exec->get_stream()>>>(result.get_data());
        result.set_executor(ref);
        return *result.get_data();
    }

    template <typename T>
    bool test_complex_is_finite_kernel()
    {
        gko::array<bool> result(exec, 1);
        test_complex_is_finite<T>
            <<<1, 1, 0, exec->get_stream()>>>(result.get_data());
        result.set_executor(ref);
        return *result.get_data();
    }
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
