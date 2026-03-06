// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

/*@GKO_PREPROCESSOR_FILENAME_HELPER@*/

#include "common/unified/matrix/amp_algorithms.hpp"

#include <iostream>
#include <limits>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


namespace gkda = gko::kernels::GKO_DEVICE_NAMESPACE::amp;

class AMPAlgorithms : public CommonTestFixture {};


// nvcc doesn't like device lambdas declared in complex classes, move it out
template <typename highest_real_type>
void bins_precision_lower_bounds(std::shared_ptr<gko::EXEC_TYPE> exec,
                                 const double rownorm, const float tol,
                                 gko::array<float>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [rownorm, tol] GKO_KERNEL(auto i, auto result) {
            const auto lbs =
                gkda::get_bins_precision_lower_bounds<highest_real_type>(
                    rownorm, tol);
            result[i] = lbs[i];
        },
        result_array.get_size(), result_array);
}

template <typename highest_real_type>
void bins_min_representable(std::shared_ptr<gko::EXEC_TYPE> exec,
                            gko::array<highest_real_type>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto result) {
            const auto mins =
                gkda::get_bins_min_representable<highest_real_type>();
            result[i] = mins[i];
        },
        result_array.get_size(), result_array);
}

template <typename highest_real_type>
void precision_bin(std::shared_ptr<gko::EXEC_TYPE> exec, const double rownorm,
                   const float tol, const highest_real_type abs_number,
                   gko::array<int>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [rownorm, tol, abs_number] GKO_KERNEL(auto i, auto result) {
            const auto lbs =
                gkda::get_bins_precision_lower_bounds<highest_real_type>(
                    rownorm, tol);
            result[i] =
                gkda::get_precision_bin<highest_real_type>(lbs, abs_number);
        },
        result_array.get_size(), result_array);
}

template <typename highest_real_type>
void adjust_bin_underflow(std::shared_ptr<gko::EXEC_TYPE> exec,
                          const highest_real_type abs_number,
                          const int initial_bin, gko::array<int>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [abs_number, initial_bin] GKO_KERNEL(auto i, auto result) {
            const auto mins =
                gkda::get_bins_min_representable<highest_real_type>();
            result[i] = gkda::adjust_bin_for_underflow<highest_real_type>(
                mins, abs_number, initial_bin);
        },
        result_array.get_size(), result_array);
}

template <typename highest_real_type>
void adjusted_bin(std::shared_ptr<gko::EXEC_TYPE> exec, const double rownorm,
                  const float tol, const highest_real_type abs_number,
                  gko::array<int>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [rownorm, tol, abs_number] GKO_KERNEL(auto i, auto result) {
            const auto lbs =
                gkda::get_bins_precision_lower_bounds<highest_real_type>(
                    rownorm, tol);
            const auto mins =
                gkda::get_bins_min_representable<highest_real_type>();
            result[i] = gkda::get_adjusted_bin<highest_real_type>(lbs, mins,
                                                                  abs_number);
        },
        result_array.get_size(), result_array);
}

void assign_to_tuple(std::shared_ptr<gko::EXEC_TYPE> exec,
                     gko::array<double>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto result) {
            auto t = std::make_tuple(3.0, -2.0f, -3, 'd');
            gkda::assign_value_to_tuple<0>(t, 1.0f, 1);
            gkda::assign_value_to_tuple<0>(t, 5, 2);
            gkda::assign_value_to_tuple<0>(t, 5, 3);
            // should do nothing:
            gkda::assign_value_to_tuple<0>(t, -6.4, 10);
            gkda::assign_value_to_tuple<0>(t, 'y', -2);
            result[0] = static_cast<double>(std::get<0>(t));
            result[1] = static_cast<double>(std::get<1>(t));
            result[2] = static_cast<double>(std::get<2>(t));
            result[3] = static_cast<double>(std::get<3>(t));
        },
        gko::size_type{1}, result_array);
}

void assign_to_array_tuple(std::shared_ptr<gko::EXEC_TYPE> exec,
                           gko::array<double>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto result) {
            double arr0[3] = {0.0, 0.0, 0.0};
            float arr1[3] = {0.0f, 0.0f, 0.0f};
            int arr2[3] = {0, 0, 0};
            auto t = std::make_tuple(arr0, arr1, arr2);
            gkda::assign_value_to_array_tuple<0>(t, 1.5, 0, 0);
            gkda::assign_value_to_array_tuple<0>(t, 2.5, 0, 2);
            gkda::assign_value_to_array_tuple<0>(t, 3.5f, 1, 1);
            gkda::assign_value_to_array_tuple<0>(t, 42, 2, 0);
            result[0] = arr0[0];
            result[1] = arr0[1];
            result[2] = arr0[2];
            result[3] = static_cast<double>(arr1[0]);
            result[4] = static_cast<double>(arr1[1]);
            result[5] = static_cast<double>(arr1[2]);
            result[6] = static_cast<double>(arr2[0]);
            result[7] = static_cast<double>(arr2[1]);
            result[8] = static_cast<double>(arr2[2]);
        },
        gko::size_type{1}, result_array);
}

void assign_to_array_tuple_oob(std::shared_ptr<gko::EXEC_TYPE> exec,
                               gko::array<double>& result_array)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto result) {
            double arr0[2] = {1.0, 2.0};
            float arr1[2] = {3.0f, 4.0f};
            auto t = std::make_tuple(arr0, arr1);
            gkda::assign_value_to_array_tuple<0>(t, 99.0, 5, 0);
            gkda::assign_value_to_array_tuple<0>(t, 99.0, -1, 0);
            result[0] = arr0[0];
            result[1] = arr0[1];
            result[2] = static_cast<double>(arr1[0]);
            result[3] = static_cast<double>(arr1[1]);
        },
        gko::size_type{1}, result_array);
}

namespace gkda = gko::kernels::GKO_DEVICE_NAMESPACE::amp;

#if GKO_AMP_HALF_IS_FP16 || GKO_AMP_HALF_IS_BFLOAT16

TEST_F(AMPAlgorithms, GetsCorrectBinLowerBoundsByPrecisionStartingDouble)
{
    const int sz = 3;
    const double rownorm = 1.0;
    const float tol = 1e-10;
    gko::array<float> result_arr(exec, sz);
    gko::array<float> expected_arr(ref, sz);
    auto expect = expected_arr.get_data();
    expect[0] = rownorm * tol / std::numeric_limits<float>::epsilon();
    expect[1] =
        rownorm * tol /
        static_cast<float>(gko::device_numeric_limits<gkda::half>::epsilon());
    expect[2] = rownorm * tol;

    bins_precision_lower_bounds<double>(exec, rownorm, tol, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectBinLowerBoundsByPrecisionStartingFloat)
{
    const int sz = 2;
    const float rownorm = 1.0;
    const float tol = 1e-6;
    gko::array<float> result_arr(exec, sz);
    gko::array<float> expected_arr(ref, sz);
    auto expect = expected_arr.get_data();
    expect[0] =
        rownorm * tol /
        static_cast<float>(gko::device_numeric_limits<gkda::half>::epsilon());
    expect[1] = rownorm * tol;

    bins_precision_lower_bounds<float>(exec, rownorm, tol, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectBinMinRepresentableStartingDouble)
{
    const int sz = 3;
    gko::array<double> result_arr(exec, sz);
    gko::array<double> expected_arr(ref, sz);
    auto expect = expected_arr.get_data();
    expect[0] = std::numeric_limits<double>::min();
    expect[1] = std::numeric_limits<float>::min();
    expect[2] =
        static_cast<double>(gko::device_numeric_limits<gkda::half>::min());

    bins_min_representable<double>(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectBinMinRepresentableStartingFloat)
{
    const int sz = 2;
    gko::array<float> result_arr(exec, sz);
    gko::array<float> expected_arr(ref, sz);
    auto expect = expected_arr.get_data();
    expect[0] = std::numeric_limits<float>::min();
    expect[1] =
        static_cast<float>(gko::device_numeric_limits<gkda::half>::min());

    bins_min_representable<float>(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectPrecisionBinDouble)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds to use as test inputs
    gko::array<float> lbs_arr(exec, 3);
    bins_precision_lower_bounds<double>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    // Value larger than lb[0] goes to bin 0
    expected_arr.get_data()[0] = 0;
    precision_bin<double>(exec, rownorm, tol, lb[0] * 2.0, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value between lb[0] and lb[1] goes to bin 1
    expected_arr.get_data()[0] = 1;
    precision_bin<double>(exec, rownorm, tol, std::sqrt(double{lb[0]} * lb[1]),
                          result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value between lb[1] and lb[2] goes to bin 2
    expected_arr.get_data()[0] = 2;
    precision_bin<double>(exec, rownorm, tol, (double{lb[1]} + lb[2]) / 2.0,
                          result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value smaller than lb[2] gets dropped
    expected_arr.get_data()[0] = -1;
    precision_bin<double>(exec, rownorm, tol, lb[2] * 0.5, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Starting from bin 1 should skip bin 0
    // expected_arr.get_data()[0] = 1;
    // precision_bin<double>(exec, rownorm, tol, lb[0] * 2.0, 1, result_arr);
    // GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectPrecisionBinFloat)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds
    gko::array<float> lbs_arr(exec, 2);
    bins_precision_lower_bounds<float>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    expected_arr.get_data()[0] = 0;
    precision_bin<float>(exec, rownorm, tol, lb[0] * 2.0f, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    expected_arr.get_data()[0] = -1;
    precision_bin<float>(exec, rownorm, tol, lb[1] * 0.5f, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, AdjustsBinForUnderflowDouble)
{
    // Get device-computed min representable values
    gko::array<double> mins_arr(exec, 3);
    bins_min_representable<double>(exec, mins_arr);
    mins_arr.set_executor(ref);
    const auto mins = mins_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    // Value representable in bin 2 stays in bin 2
    expected_arr.get_data()[0] = 2;
    adjust_bin_underflow<double>(exec, mins[2] * 2.0, 2, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value below min of bin 2 moves to higher-precision bin
    const double val_underflow_half = mins[2] * 0.5;
#if GKO_AMP_HALF_IS_FP16
    expected_arr.get_data()[0] = 1;
#else
    expected_arr.get_data()[0] = 0;
#endif
    adjust_bin_underflow<double>(exec, val_underflow_half, 2, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value below min of bin 1 moves to bin 0
    expected_arr.get_data()[0] = 0;
    adjust_bin_underflow<double>(exec, mins[1] * 0.5, 2, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Dropped values stay dropped
    expected_arr.get_data()[0] = -1;
    adjust_bin_underflow<double>(exec, 1e-100, -1, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Bin 0 stays at bin 0 even for tiny values
    expected_arr.get_data()[0] = 0;
    adjust_bin_underflow<double>(exec, 1e-320, 0, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, AdjustsBinForUnderflowFloat)
{
    // Get device-computed min representable values
    gko::array<float> mins_arr(exec, 2);
    bins_min_representable<float>(exec, mins_arr);
    mins_arr.set_executor(ref);
    const auto mins = mins_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    expected_arr.get_data()[0] = 1;
    adjust_bin_underflow<float>(exec, mins[1] * 2.0f, 1, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsAdjustedBinDouble)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds and min representable
    gko::array<float> lbs_arr(exec, 3);
    bins_precision_lower_bounds<double>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<double> mins_arr(exec, 3);
    bins_min_representable<double>(exec, mins_arr);
    mins_arr.set_executor(ref);
    const auto mins = mins_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    // Large value goes to bin 0
    expected_arr.get_data()[0] = 0;
    adjusted_bin<double>(exec, rownorm, tol, lb[0] * 2.0, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Values just smaller than half min
    const double val_under = mins[2] / 1.1;
#if GKO_AMP_HALF_IS_FP16
    expected_arr.get_data()[0] = 1;
#else
    expected_arr.get_data()[0] = -1;
#endif
    adjusted_bin<double>(exec, rownorm, tol, val_under, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Very small value gets dropped
    expected_arr.get_data()[0] = -1;
    adjusted_bin<double>(exec, rownorm, tol, lb[2] * 0.5, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsAdjustedBinFloat)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds
    gko::array<float> lbs_arr(exec, 2);
    bins_precision_lower_bounds<float>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    expected_arr.get_data()[0] = 0;
    adjusted_bin<float>(exec, rownorm, tol, lb[0] * 2.0f, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    expected_arr.get_data()[0] = -1;
    adjusted_bin<float>(exec, rownorm, tol, lb[1] * 0.5f, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

#else

TEST_F(AMPAlgorithms, GetsCorrectBinLowerBoundsByPrecisionStartingDouble)
{
    const int sz = 2;
    const double rownorm = 1.0;
    const float tol = 1e-10;
    gko::array<float> result_arr(exec, sz);
    gko::array<float> expected_arr(ref, sz);
    auto expect = expected_arr.get_data();
    expect[0] = rownorm * tol / std::numeric_limits<float>::epsilon();
    expect[1] = rownorm * tol;

    bins_precision_lower_bounds<double>(exec, rownorm, tol, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectBinLowerBoundsByPrecisionStartingFloat)
{
    const int sz = 1;
    const float rownorm = 1.0;
    const float tol = 1e-10;
    gko::array<float> result_arr(exec, sz);
    gko::array<float> expected_arr(ref, sz);
    expected_arr.get_data()[0] = rownorm * tol;

    bins_precision_lower_bounds<float>(exec, rownorm, tol, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectBinMinRepresentableStartingDouble)
{
    const int sz = 2;
    gko::array<double> result_arr(exec, sz);
    gko::array<double> expected_arr(ref, sz);
    auto expect = expected_arr.get_data();
    expect[0] = std::numeric_limits<double>::min();
    expect[1] = static_cast<double>(std::numeric_limits<float>::min());

    bins_min_representable<double>(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectBinMinRepresentableStartingFloat)
{
    const int sz = 1;
    gko::array<float> result_arr(exec, sz);
    gko::array<float> expected_arr(ref, sz);
    expected_arr.get_data()[0] = std::numeric_limits<float>::min();

    bins_min_representable<float>(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectPrecisionBinDouble)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds
    gko::array<float> lbs_arr(exec, 2);
    bins_precision_lower_bounds<double>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    // Value larger than lb[0] goes to bin 0
    expected_arr.get_data()[0] = 0;
    precision_bin<double>(exec, rownorm, tol, lb[0] * 2.0, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value between lb[0] and lb[1] goes to bin 1
    expected_arr.get_data()[0] = 1;
    precision_bin<double>(exec, rownorm, tol, (double{lb[0]} + lb[1]) / 2.0,
                          result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value smaller than lb[1] gets dropped
    expected_arr.get_data()[0] = -1;
    precision_bin<double>(exec, rownorm, tol, lb[1] * 0.5, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Starting from bin 1 should skip bin 0
    // expected_arr.get_data()[0] = 1;
    // precision_bin<double>(exec, rownorm, tol, lb[0] * 2.0, 1, result_arr);
    // GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsCorrectPrecisionBinFloat)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds
    gko::array<float> lbs_arr(exec, 1);
    bins_precision_lower_bounds<float>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    expected_arr.get_data()[0] = 0;
    precision_bin<float>(exec, rownorm, tol, lb[0] * 2.0f, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    expected_arr.get_data()[0] = -1;
    precision_bin<float>(exec, rownorm, tol, lb[0] * 0.5f, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, AdjustsBinForUnderflowDouble)
{
    // Get device-computed min representable values
    gko::array<double> mins_arr(exec, 2);
    bins_min_representable<double>(exec, mins_arr);
    mins_arr.set_executor(ref);
    const auto mins = mins_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    // Value representable in bin 1 stays in bin 1
    expected_arr.get_data()[0] = 1;
    adjust_bin_underflow<double>(exec, mins[1] * 2.0, 1, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value below min of bin 1 moves to bin 0
    expected_arr.get_data()[0] = 0;
    adjust_bin_underflow<double>(exec, mins[1] * 0.5, 1, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Dropped values stay dropped
    expected_arr.get_data()[0] = -1;
    adjust_bin_underflow<double>(exec, 1e-100, -1, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Bin 0 stays at bin 0 even for tiny values
    expected_arr.get_data()[0] = 0;
    adjust_bin_underflow<double>(exec, 1e-320, 0, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, GetsAdjustedBinDouble)
{
    const double rownorm = 1.0;
    const float tol = 1e-10;
    // Get device-computed lower bounds and min representable
    gko::array<float> lbs_arr(exec, 2);
    bins_precision_lower_bounds<double>(exec, rownorm, tol, lbs_arr);
    lbs_arr.set_executor(ref);
    const auto lb = lbs_arr.get_const_data();
    gko::array<double> mins_arr(exec, 2);
    bins_min_representable<double>(exec, mins_arr);
    mins_arr.set_executor(ref);
    const auto mins = mins_arr.get_const_data();
    gko::array<int> result_arr(exec, 1);
    gko::array<int> expected_arr(ref, 1);

    // Large value goes to bin 0
    expected_arr.get_data()[0] = 0;
    adjusted_bin<double>(exec, rownorm, tol, lb[0] * 2.0, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);

    // Value that would go to bin 1 and is representable stays in bin 1
    const double val_bin1 = (double{lb[0]} + lb[1]) / 2.0;
    if (val_bin1 >= mins[1]) {
        expected_arr.get_data()[0] = 1;
        adjusted_bin<double>(exec, rownorm, tol, val_bin1, result_arr);
        GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
    }

    // Very small value gets dropped
    expected_arr.get_data()[0] = -1;
    adjusted_bin<double>(exec, rownorm, tol, lb[1] * 0.5, result_arr);
    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

#endif  // GKO_AMP_HALF_IS_FP16 || GKO_AMP_HALF_IS_BFLOAT16


TEST_F(AMPAlgorithms, AssignsValueToTuple)
{
    gko::array<double> result_arr(exec, 4);
    gko::array<double> expected_arr(ref, 4);
    auto expect = expected_arr.get_data();
    expect[0] = 3.0;
    expect[1] = 1.0;
    expect[2] = 5.0;
    expect[3] = 5.0;

    assign_to_tuple(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, AssignsValueToArrayTuple)
{
    gko::array<double> result_arr(exec, 9);
    gko::array<double> expected_arr(ref, 9);
    auto expect = expected_arr.get_data();
    expect[0] = 1.5;
    expect[1] = 0.0;
    expect[2] = 2.5;
    expect[3] = 0.0;
    expect[4] = 3.5;
    expect[5] = 0.0;
    expect[6] = 42.0;
    expect[7] = 0.0;
    expect[8] = 0.0;

    assign_to_array_tuple(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}

TEST_F(AMPAlgorithms, AssignsValueToArrayTupleWithOutOfRangeIndex)
{
    gko::array<double> result_arr(exec, 4);
    gko::array<double> expected_arr(ref, 4);
    auto expect = expected_arr.get_data();
    expect[0] = 1.0;
    expect[1] = 2.0;
    expect[2] = 3.0;
    expect[3] = 4.0;

    assign_to_array_tuple_oob(exec, result_arr);

    GKO_ASSERT_ARRAY_EQ(result_arr, expected_arr);
}
