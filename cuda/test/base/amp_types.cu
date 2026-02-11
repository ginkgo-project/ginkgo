// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/base/amp_types.hpp"

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "common/cuda_hip/base/types.hpp"
#include "cuda/test/utils.hpp"


template <typename T, typename U>
__global__ void test_types_are_same(bool* result)
{
    *result = std::is_same<T, U>::value;
}


template <typename T>
__global__ void test_precision_index(bool* result, const int expected)
{
    *result = (gko::amp::precision_index<T>::index == expected);
}


template <int i, typename RealOrComplexType, typename Expected>
__global__ void test_type_at_idx(bool* result)
{
    *result = std::is_same<gko::amp::type_at_idx<i, RealOrComplexType>,
                           Expected>::value;
}


template <typename HighestType, typename Expected>
__global__ void test_narrow_types(bool* result)
{
    *result = std::is_same<typename gko::amp::narrow_types<HighestType>::type,
                           Expected>::value;
}


class AMPTypes : public CudaTestFixture {};

#ifdef GINKGO_HAVE_AMP_HALF

#ifdef GKO_AMP_HALF_IS_BFLOAT16
using testhalf = __nv_bfloat16;
#else
using testhalf = __half;
#endif

TEST_F(AMPTypes, SupportsCorrectPrecisions)
{
    using CorrectTypes = std::tuple<double, float, testhalf>;
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::supported_precisions, CorrectTypes>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, SupportsCorrectRealTypes)
{
    static_assert(std::is_same_v<gko::amp::half, testhalf>, "Wrong half!");
    using CorrectTypes = std::tuple<double, float, testhalf>;
    static_assert(
        std::is_same_v<CorrectTypes, gko::amp::supported_types<float>::type>,
        "Wrong types on host!");
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::supported_types<float>::type, CorrectTypes>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, SupportsCorrectComplexTypes)
{
    using CorrectTypes =
        std::tuple<thrust::complex<double>, thrust::complex<float>,
                   thrust::complex<testhalf>>;
    static_assert(
        std::is_same_v<CorrectTypes,
                       gko::amp::supported_types<thrust::complex<float>>::type>,
        "Wrong types on host!");
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::supported_types<thrust::complex<float>>::type,
                        CorrectTypes>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, DeviceToComplexMapsTuple)
{
    using Expected = std::tuple<thrust::complex<double>, thrust::complex<float>,
                                thrust::complex<testhalf>>;
    static_assert(
        std::is_same_v<
            gko::amp::device_to_complex<gko::amp::supported_precisions>,
            Expected>,
        "device_to_complex should map tuple to complex");
    gko::array<bool> result(exec, 1);

    test_types_are_same<
        gko::amp::device_to_complex<gko::amp::supported_precisions>, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NumAmpPrecisionsIsCorrect)
{
    static_assert(gko::amp::num_amp_precisions == 3, "should be 3 with half");
    gko::array<bool> result(exec, 1);

    test_types_are_same<
        std::integral_constant<int, gko::amp::num_amp_precisions>,
        std::integral_constant<int, 3>>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, RealTypeAtIdxIsCorrect)
{
    static_assert(std::is_same_v<gko::amp::real_type_at_idx<0>, double>);
    static_assert(std::is_same_v<gko::amp::real_type_at_idx<1>, float>);
    static_assert(std::is_same_v<gko::amp::real_type_at_idx<2>, testhalf>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::real_type_at_idx<2>, testhalf>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, TypeAtIdxRealIsCorrect)
{
    static_assert(std::is_same_v<gko::amp::type_at_idx<0, double>, double>);
    static_assert(std::is_same_v<gko::amp::type_at_idx<1, double>, float>);
    static_assert(std::is_same_v<gko::amp::type_at_idx<2, double>, testhalf>);
    gko::array<bool> result(exec, 1);

    test_type_at_idx<2, double, testhalf>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, TypeAtIdxComplexIsCorrect)
{
    using cx_double = thrust::complex<double>;
    using cx_float = thrust::complex<float>;
    using cx_half = thrust::complex<testhalf>;
    static_assert(
        std::is_same_v<gko::amp::type_at_idx<0, cx_double>, cx_double>);
    static_assert(
        std::is_same_v<gko::amp::type_at_idx<1, cx_double>, cx_float>);
    static_assert(std::is_same_v<gko::amp::type_at_idx<2, cx_double>, cx_half>);
    gko::array<bool> result(exec, 1);

    test_type_at_idx<2, cx_double, cx_half>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, PrecisionIndexIsCorrect)
{
    static_assert(gko::amp::precision_index<double>::index == 0);
    static_assert(gko::amp::precision_index<float>::index == 1);
    static_assert(gko::amp::precision_index<testhalf>::index == 2);
    static_assert(gko::amp::precision_index<thrust::complex<double>>::index ==
                  0);
    static_assert(gko::amp::precision_index<thrust::complex<float>>::index ==
                  1);
    static_assert(gko::amp::precision_index<thrust::complex<testhalf>>::index ==
                  2);
    gko::array<bool> result(exec, 1);

    test_precision_index<testhalf>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data(), 2);

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesFromDoubleIsCorrect)
{
    using Expected = std::tuple<double, float, testhalf>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<double>::type, Expected>);
    static_assert(gko::amp::narrow_types<double>::num_types == 3);
    gko::array<bool> result(exec, 1);

    test_narrow_types<double, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesFromFloatIsCorrect)
{
    using Expected = std::tuple<float, testhalf>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<float>::type, Expected>);
    static_assert(gko::amp::narrow_types<float>::num_types == 2);
    gko::array<bool> result(exec, 1);

    test_narrow_types<float, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesFromHalfIsCorrect)
{
    using Expected = std::tuple<testhalf>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<testhalf>::type, Expected>);
    static_assert(gko::amp::narrow_types<testhalf>::num_types == 1);
    gko::array<bool> result(exec, 1);

    test_narrow_types<testhalf, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesComplexFromDoubleIsCorrect)
{
    using cx_double = thrust::complex<double>;
    using cx_float = thrust::complex<float>;
    using cx_half = thrust::complex<testhalf>;
    using Expected = std::tuple<cx_double, cx_float, cx_half>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<cx_double>::type, Expected>);
    static_assert(gko::amp::narrow_types<cx_double>::num_types == 3);
    gko::array<bool> result(exec, 1);

    test_narrow_types<cx_double, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesComplexFromHalfIsCorrect)
{
    using cx_half = thrust::complex<testhalf>;
    using Expected = std::tuple<cx_half>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<cx_half>::type, Expected>);
    static_assert(gko::amp::narrow_types<cx_half>::num_types == 1);
    gko::array<bool> result(exec, 1);

    test_narrow_types<cx_half, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, PrecisionArrayHasCorrectSize)
{
    static_assert(std::is_same_v<gko::amp::precision_array<int, double>,
                                 std::array<int, 3>>);
    static_assert(std::is_same_v<gko::amp::precision_array<int, float>,
                                 std::array<int, 2>>);
    static_assert(std::is_same_v<gko::amp::precision_array<int, testhalf>,
                                 std::array<int, 1>>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::precision_array<int, double>,
                        std::array<int, 3>>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

#else

// test only double and float

TEST_F(AMPTypes, SupportsCorrectPrecisions)
{
    using CorrectTypes = std::tuple<double, float>;
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::supported_precisions, CorrectTypes>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, DeviceToComplexMapsTuple)
{
    using Expected =
        std::tuple<thrust::complex<double>, thrust::complex<float>>;
    static_assert(
        std::is_same_v<
            gko::amp::device_to_complex<gko::amp::supported_precisions>,
            Expected>,
        "device_to_complex should map tuple to complex");
    gko::array<bool> result(exec, 1);

    test_types_are_same<
        gko::amp::device_to_complex<gko::amp::supported_precisions>, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NumAmpPrecisionsIsCorrect)
{
    static_assert(gko::amp::num_amp_precisions == 2,
                  "should be 2 without half");
    gko::array<bool> result(exec, 1);

    test_types_are_same<
        std::integral_constant<int, gko::amp::num_amp_precisions>,
        std::integral_constant<int, 2>>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, RealTypeAtIdxIsCorrect)
{
    static_assert(std::is_same_v<gko::amp::real_type_at_idx<0>, double>);
    static_assert(std::is_same_v<gko::amp::real_type_at_idx<1>, float>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::real_type_at_idx<1>, float>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, TypeAtIdxRealIsCorrect)
{
    static_assert(std::is_same_v<gko::amp::type_at_idx<0, double>, double>);
    static_assert(std::is_same_v<gko::amp::type_at_idx<1, double>, float>);
    gko::array<bool> result(exec, 1);

    test_type_at_idx<1, double, float>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, TypeAtIdxComplexIsCorrect)
{
    using cx_double = thrust::complex<double>;
    using cx_float = thrust::complex<float>;
    static_assert(
        std::is_same_v<gko::amp::type_at_idx<0, cx_double>, cx_double>);
    static_assert(
        std::is_same_v<gko::amp::type_at_idx<1, cx_double>, cx_float>);
    gko::array<bool> result(exec, 1);

    test_type_at_idx<1, cx_double, cx_float>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, PrecisionIndexIsCorrect)
{
    static_assert(gko::amp::precision_index<double>::index == 0);
    static_assert(gko::amp::precision_index<float>::index == 1);
    static_assert(gko::amp::precision_index<thrust::complex<double>>::index ==
                  0);
    static_assert(gko::amp::precision_index<thrust::complex<float>>::index ==
                  1);
    gko::array<bool> result(exec, 1);

    test_precision_index<float>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data(), 1);

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesFromDoubleIsCorrect)
{
    using Expected = std::tuple<double, float>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<double>::type, Expected>);
    static_assert(gko::amp::narrow_types<double>::num_types == 2);
    gko::array<bool> result(exec, 1);

    test_narrow_types<double, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesFromFloatIsCorrect)
{
    using Expected = std::tuple<float>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<float>::type, Expected>);
    static_assert(gko::amp::narrow_types<float>::num_types == 1);
    gko::array<bool> result(exec, 1);

    test_narrow_types<float, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesComplexFromDoubleIsCorrect)
{
    using cx_double = thrust::complex<double>;
    using cx_float = thrust::complex<float>;
    using Expected = std::tuple<cx_double, cx_float>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<cx_double>::type, Expected>);
    static_assert(gko::amp::narrow_types<cx_double>::num_types == 2);
    gko::array<bool> result(exec, 1);

    test_narrow_types<cx_double, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, NarrowTypesComplexFromFloatIsCorrect)
{
    using cx_float = thrust::complex<float>;
    using Expected = std::tuple<cx_float>;
    static_assert(
        std::is_same_v<gko::amp::narrow_types<cx_float>::type, Expected>);
    static_assert(gko::amp::narrow_types<cx_float>::num_types == 1);
    gko::array<bool> result(exec, 1);

    test_narrow_types<cx_float, Expected>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, PrecisionArrayHasCorrectSize)
{
    static_assert(std::is_same_v<gko::amp::precision_array<int, double>,
                                 std::array<int, 2>>);
    static_assert(std::is_same_v<gko::amp::precision_array<int, float>,
                                 std::array<int, 1>>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::precision_array<int, double>,
                        std::array<int, 2>>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

#endif


// Tests independent of half support

TEST_F(AMPTypes, DeviceToComplexMapsRealToComplex)
{
    static_assert(std::is_same_v<gko::amp::device_to_complex<float>,
                                 thrust::complex<float>>);
    static_assert(std::is_same_v<gko::amp::device_to_complex<double>,
                                 thrust::complex<double>>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::device_to_complex<double>,
                        thrust::complex<double>>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, DeviceToComplexPreservesComplex)
{
    static_assert(
        std::is_same_v<gko::amp::device_to_complex<thrust::complex<float>>,
                       thrust::complex<float>>);
    static_assert(
        std::is_same_v<gko::amp::device_to_complex<thrust::complex<double>>,
                       thrust::complex<double>>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::device_to_complex<thrust::complex<double>>,
                        thrust::complex<double>>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, DeviceRemoveComplexOnReal)
{
    static_assert(
        std::is_same_v<gko::amp::device_remove_complex<float>, float>);
    static_assert(
        std::is_same_v<gko::amp::device_remove_complex<double>, double>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<gko::amp::device_remove_complex<double>, double>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}

TEST_F(AMPTypes, DeviceRemoveComplexOnComplex)
{
    static_assert(
        std::is_same_v<gko::amp::device_remove_complex<thrust::complex<float>>,
                       float>);
    static_assert(
        std::is_same_v<gko::amp::device_remove_complex<thrust::complex<double>>,
                       double>);
    gko::array<bool> result(exec, 1);

    test_types_are_same<
        gko::amp::device_remove_complex<thrust::complex<double>>, double>
        <<<1, 1, 0, exec->get_stream()>>>(result.get_data());

    result.set_executor(ref);
    EXPECT_TRUE(result.get_data()[0]);
}


}  // namespace gko
