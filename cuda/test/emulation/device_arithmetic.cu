// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/emulation/device_arithmetic.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include <cuda_fp16.h>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/matrix/device_views.hpp>

#include "accessor/cuda_helper.hpp"
#include "accessor/range.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "common/cuda_hip/base/thrust_macro.hpp"
#include "common/unified/base/arithmetic_accessor.hpp"
#include "cuda/test/utils.hpp"


namespace {


#if GINKGO_EMULATE_DOUBLE
using double_arithmetic = gko::emulated_double;
#else
using double_arithmetic = double;
#endif


template <typename Arithmetic, typename Storage>
using host_range =
    gko::acc::range<gko::acc::reduced_row_major<2, Arithmetic, Storage>>;

template <typename Arithmetic, typename Storage>
using device_accessor = typename decltype(gko::acc::as_cuda_range(
    std::declval<host_range<Arithmetic, Storage>>()))::accessor;

// one scalar per column
template <typename Arithmetic, typename Storage>
using host_scaled_range = gko::acc::range<
    gko::acc::scaled_reduced_row_major<2, Arithmetic, Storage, 0b01>>;


TEST(DeviceArithmetic, ReplacesOnlyDouble)
{
    static_assert(
        std::is_same_v<gko::device_arithmetic_t<double>, double_arithmetic>);
    static_assert(std::is_same_v<gko::device_arithmetic_t<const double>,
                                 double_arithmetic>);
    static_assert(std::is_same_v<gko::device_arithmetic_t<float>, float>);
    static_assert(
        std::is_same_v<gko::device_arithmetic_t<gko::half>, gko::half>);
    static_assert(std::is_same_v<gko::device_arithmetic_t<std::complex<double>>,
                                 std::complex<double>>);
}


TEST(AsCudaRange, KeepsDeviceArithmeticType)
{
    using accessor = device_accessor<double_arithmetic, const float>;

    static_assert(
        std::is_same_v<typename accessor::arithmetic_type, double_arithmetic>);
    static_assert(std::is_same_v<typename accessor::storage_type, const float>);
}


TEST(AsCudaRange, MapsToCudaTypes)
{
    static_assert(
        std::is_same_v<
            typename device_accessor<gko::half, gko::half>::arithmetic_type,
            __half>);
    static_assert(std::is_same_v<
                  typename device_accessor<gko::half, gko::half>::storage_type,
                  __half>);
    static_assert(
        std::is_same_v<
            typename device_accessor<std::complex<double>,
                                     std::complex<double>>::arithmetic_type,
            GKO_THRUST_QUALIFIER::complex<double>>);
}


TEST(Arithmetic, ReplacesArithmeticTypeOfReducedRangeButNotStorageType)
{
    using accessor = typename decltype(gko::kernels::cuda::arithmetic<double>(
        std::declval<host_range<double, const float>>()))::accessor;

    static_assert(
        std::is_same_v<typename accessor::arithmetic_type, double_arithmetic>);
    static_assert(std::is_same_v<typename accessor::storage_type, const float>);
}


TEST(Arithmetic, ReplacesArithmeticTypeOfScaledRangeButNotScalarType)
{
    using accessor = typename decltype(gko::kernels::cuda::arithmetic<double>(
        std::declval<
            host_scaled_range<double, const std::int32_t>>()))::accessor;

    static_assert(
        std::is_same_v<typename accessor::arithmetic_type, double_arithmetic>);
    static_assert(
        std::is_same_v<typename accessor::storage_type, const std::int32_t>);
    static_assert(std::is_same_v<typename accessor::scalar_type, const double>);
}


TEST(Arithmetic, CreatesRowMajorRangeWithoutConversion)
{
    std::array<float, 3> data{};

    auto range = gko::kernels::cuda::arithmetic<float>(data.data(), 3);
    auto device_range = gko::acc::as_cuda_range(range);

    static_assert(
        std::is_same_v<decltype(range),
                       gko::acc::range<gko::acc::row_major<float, 1>>>);
    static_assert(std::is_same_v<decltype(device_range), decltype(range)>);
    static_assert(std::is_same_v<
                  gko::kernels::cuda::arithmetic_type_t<decltype(device_range)>,
                  float>);
    EXPECT_EQ(device_range.length(0), 3);
    ASSERT_EQ(&device_range[1], data.data() + 1);
}


TEST(Arithmetic, ComputesDoubleInDeviceArithmetic)
{
    std::array<double, 3> data{};

    auto range = gko::kernels::cuda::arithmetic<double>(
        static_cast<const double*>(data.data()), 3);
    auto device_range = gko::acc::as_cuda_range(range);

#if GINKGO_EMULATE_DOUBLE
    static_assert(std::is_same_v<decltype(range),
                                 gko::acc::range<gko::acc::reduced_row_major<
                                     1, gko::emulated_double, const double>>>);
#else
    static_assert(
        std::is_same_v<decltype(range),
                       gko::acc::range<gko::acc::row_major<const double, 1>>>);
#endif
    static_assert(std::is_same_v<decltype(device_range), decltype(range)>);
    static_assert(std::is_same_v<
                  gko::kernels::cuda::arithmetic_type_t<decltype(device_range)>,
                  double_arithmetic>);
    ASSERT_EQ(device_range.length(0), 3);
}


TEST(Arithmetic, ConvertsStorageOfOtherPrecision)
{
    using range = decltype(gko::kernels::cuda::arithmetic<double>(
        std::declval<const float*>(), 0));
    using device_range =
        decltype(gko::acc::as_cuda_range(std::declval<range>()));

    static_assert(
        std::is_same_v<range, gko::acc::range<gko::acc::reduced_row_major<
                                  1, double_arithmetic, const float>>>);
    static_assert(std::is_same_v<device_range, range>);
}


TEST(Arithmetic, MapsToCudaTypes)
{
    using half_range = decltype(gko::acc::as_cuda_range(
        gko::kernels::cuda::arithmetic<gko::half>(
            std::declval<const gko::half*>(), 0)));
    using complex_range = decltype(gko::acc::as_cuda_range(
        gko::kernels::cuda::arithmetic<std::complex<double>>(
            std::declval<const std::complex<double>*>(), 0)));

    static_assert(
        std::is_same_v<half_range,
                       gko::acc::range<gko::acc::row_major<const __half, 1>>>);
    static_assert(
        std::is_same_v<complex_range,
                       gko::acc::range<gko::acc::row_major<
                           const GKO_THRUST_QUALIFIER::complex<double>, 1>>>);
}


TEST(Arithmetic, CreatesTwoDimensionalRangeFromDenseMatrix)
{
    std::array<float, 8> data{};
    gko::matrix::view::dense<float> mtx{gko::dim<2>{2, 3}, 4, data.data()};

    auto range = gko::kernels::cuda::arithmetic<float>(mtx);

    static_assert(
        std::is_same_v<decltype(range),
                       gko::acc::range<gko::acc::row_major<float, 2>>>);
    EXPECT_EQ(range.length(0), 2);
    EXPECT_EQ(range.length(1), 3);
    ASSERT_EQ(&range(1, 2), data.data() + 6);
}


TEST(Arithmetic, CreatesOneDimensionalRangeFromCsrValues)
{
    std::array<float, 4> values{};
    gko::matrix::view::csr<float, gko::int32> mtx{
        gko::dim<2>{2, 3}, 4, values.data(), nullptr, nullptr};

    auto range = gko::kernels::cuda::arithmetic<float>(mtx);

    static_assert(
        std::is_same_v<decltype(range),
                       gko::acc::range<gko::acc::row_major<float, 1>>>);
    EXPECT_EQ(range.length(0), 4);
    ASSERT_EQ(&range[3], values.data() + 3);
}


TEST(Arithmetic, CreatesOneDimensionalRangeFromDenseValues)
{
    std::array<float, 8> data{};
    gko::matrix::view::dense<float> mtx{gko::dim<2>{2, 3}, 4, data.data()};

    auto range = gko::kernels::cuda::arithmetic<float, 1>(mtx);

    static_assert(
        std::is_same_v<decltype(range),
                       gko::acc::range<gko::acc::row_major<float, 1>>>);
    EXPECT_EQ(range.length(0), 7);
    ASSERT_EQ(&range[6], data.data() + 6);
}


TEST(Arithmetic, CreatesRowMajorRangeOverDeviceTypesWithoutConversion)
{
    using gko::kernels::cuda::arithmetic;
    using complex_type = gko::kernels::cuda::device_type<std::complex<double>>;
    using half_type = gko::kernels::cuda::device_type<gko::half>;

    static_assert(std::is_same_v<
                  decltype(arithmetic<std::complex<double>>(
                      std::declval<const complex_type*>(), 1)),
                  gko::acc::range<gko::acc::row_major<const complex_type, 1>>>);
    static_assert(
        std::is_same_v<decltype(arithmetic<gko::half>(
                           std::declval<half_type*>(), 1)),
                       gko::acc::range<gko::acc::row_major<half_type, 1>>>);
}


TEST(ArithmeticType, OfPointerIsItsValueType)
{
    static_assert(
        std::is_same_v<gko::kernels::cuda::arithmetic_type_t<const double*>,
                       double>);
    static_assert(
        std::is_same_v<gko::kernels::cuda::arithmetic_type_t<float* const>,
                       float>);
}


TEST(Tolerance, UsesPrecisionOfDeviceArithmetic)
{
#if GINKGO_EMULATE_DOUBLE
    // two float words hold 2 * 24 significand bits
    EXPECT_DOUBLE_EQ(r<double>::value, 10 * 0x1p-47);
#else
    EXPECT_DOUBLE_EQ(r<double>::value,
                     10 * std::numeric_limits<double>::epsilon());
#endif
    EXPECT_FLOAT_EQ(r<float>::value,
                    10 * std::numeric_limits<float>::epsilon());
    // complex<double> is computed in itself
    EXPECT_DOUBLE_EQ(r<std::complex<double>>::value,
                     10 * std::numeric_limits<double>::epsilon() * 1.4142);
}


template <typename InputAccessor, typename OutputAccessor>
__global__ void add_product(gko::acc::range<InputAccessor> a,
                            gko::acc::range<InputAccessor> b,
                            gko::acc::range<OutputAccessor> c)
{
    static_assert(std::is_same_v<typename OutputAccessor::arithmetic_type,
                                 double_arithmetic>);
    const auto row = threadIdx.x;
    const auto col = threadIdx.y;
    // reads a and b, reads and writes c, all in the arithmetic type
    c(row, col) += a(row, col) * b(row, col);
}


class AccessorKernel : public CudaTestFixture {};


TEST_F(AccessorKernel, ComputesInDeviceArithmetic)
{
    constexpr int rows = 4;
    constexpr int cols = 3;
    const std::array<gko::acc::size_type, 2> size{{rows, cols}};
    gko::array<double> a{ref, rows * cols};
    gko::array<double> b{ref, rows * cols};
    gko::array<double> c{ref, rows * cols};
    // positive values that float cannot represent, so that the sum does not
    // cancel and computing in float would be visible
    for (int i = 0; i < rows * cols; ++i) {
        a.get_data()[i] = 1.0 / 3.0 + i;
        b.get_data()[i] = std::sqrt(2.0) * (i + 1);
        c.get_data()[i] = 1.0 / 7.0;
    }
    gko::array<double> d_a{exec, a};
    gko::array<double> d_b{exec, b};
    gko::array<double> d_c{exec, c};
    const auto a_range = gko::kernels::cuda::arithmetic<double>(
        host_range<double, const double>{size, d_a.get_const_data()});
    const auto b_range = gko::kernels::cuda::arithmetic<double>(
        host_range<double, const double>{size, d_b.get_const_data()});
    const auto c_range = gko::kernels::cuda::arithmetic<double>(
        host_range<double, double>{size, d_c.get_data()});

    add_product<<<1, dim3(rows, cols), 0, exec->get_stream()>>>(
        gko::acc::as_cuda_range(a_range), gko::acc::as_cuda_range(b_range),
        gko::acc::as_cuda_range(c_range));

    gko::array<double> result{ref, d_c};
    for (int i = 0; i < rows * cols; ++i) {
        const auto expected = c.get_const_data()[i] +
                              a.get_const_data()[i] * b.get_const_data()[i];
        // the tolerance of the kernel tests, which allows for the precision of
        // emulated double, but not for the one of a single float
        EXPECT_NEAR(result.get_const_data()[i], expected,
                    r<double>::value * std::abs(expected))
            << "at " << i;
    }
}


template <typename InputAccessor, typename OutputAccessor>
__global__ void copy_twice(gko::acc::range<InputAccessor> in,
                           gko::acc::range<OutputAccessor> out)
{
    static_assert(std::is_same_v<typename OutputAccessor::arithmetic_type,
                                 double_arithmetic>);
    const auto row = threadIdx.x;
    const auto col = threadIdx.y;
    const auto value = in(row, col);
    out(row, col) = value + value;
}


TEST_F(AccessorKernel, ConvertsIntegerStorageExactly)
{
    constexpr int rows = 4;
    constexpr int cols = 2;
    const std::array<gko::acc::size_type, 2> size{{rows, cols}};
    // integers with more bits than a float holds, whose halves have a
    // fraction for the conversion back to round towards zero
    const std::array<std::int32_t, rows * cols> input{
        {(1 << 30) + 3, -((1 << 30) + 3), (1 << 24) + 1, -((1 << 24) + 1),
         2147483647, -2147483647, 5, -5}};
    gko::array<std::int32_t> in{ref, input.begin(), input.end()};
    // the output stores 2 * in divided by the scalar of its column, so column
    // 0 gets in and column 1 a quarter of in, rounded towards zero
    gko::array<double> in_scalar{ref, {1.0, 1.0}};
    gko::array<double> out_scalar{ref, {2.0, 8.0}};
    gko::array<std::int32_t> d_in{exec, in};
    gko::array<std::int32_t> d_out{exec, rows * cols};
    gko::array<double> d_in_scalar{exec, in_scalar};
    gko::array<double> d_out_scalar{exec, out_scalar};
    const auto in_range = gko::kernels::cuda::arithmetic<double>(
        host_scaled_range<double, const std::int32_t>{
            size, d_in.get_const_data(), d_in_scalar.get_const_data()});
    const auto out_range = gko::kernels::cuda::arithmetic<double>(
        host_scaled_range<double, std::int32_t>{size, d_out.get_data(),
                                                d_out_scalar.get_data()});

    copy_twice<<<1, dim3(rows, cols), 0, exec->get_stream()>>>(
        gko::acc::as_cuda_range(in_range), gko::acc::as_cuda_range(out_range));

    gko::array<std::int32_t> result{ref, d_out};
    for (int i = 0; i < rows * cols; ++i) {
        // exact in double
        const auto expected = static_cast<std::int32_t>(
            2.0 * input[i] / out_scalar.get_const_data()[i % cols]);
        EXPECT_EQ(result.get_const_data()[i], expected) << "at " << i;
    }
}


TEST_F(AccessorKernel, ConvertsHalfStorageExactly)
{
    constexpr int rows = 2;
    constexpr int cols = 2;
    const std::array<gko::acc::size_type, 2> size{{rows, cols}};
    // values with all bits of the half significand
    const std::array<float, rows * cols> input{
        {0.333251953125f, -1.9990234375f, 1023.5f, -6.103515625e-05f}};
    gko::array<gko::half> in{ref, rows * cols};
    for (int i = 0; i < rows * cols; ++i) {
        in.get_data()[i] = gko::half{input[i]};
    }
    gko::array<gko::half> d_in{exec, in};
    gko::array<gko::half> d_out{exec, rows * cols};
    const auto in_range = gko::kernels::cuda::arithmetic<double>(
        host_range<double, const gko::half>{size, d_in.get_const_data()});
    const auto out_range = gko::kernels::cuda::arithmetic<double>(
        host_range<double, gko::half>{size, d_out.get_data()});

    copy_twice<<<1, dim3(rows, cols), 0, exec->get_stream()>>>(
        gko::acc::as_cuda_range(in_range), gko::acc::as_cuda_range(out_range));

    gko::array<gko::half> result{ref, d_out};
    for (int i = 0; i < rows * cols; ++i) {
        // doubling is exact in half
        EXPECT_EQ(static_cast<float>(result.get_const_data()[i]), 2 * input[i])
            << "at " << i;
    }
}


}  // namespace
