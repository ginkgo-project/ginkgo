// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/intrinsics.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "core/test/utils.hpp"
#include "test/utils/common_fixture.hpp"


class Intrinsics : public CommonTestFixture {
public:
    Intrinsics()
        : input_array32{exec, {1u, 0x80ffu, 0x100u, ~0u}},
          input_array64{exec, {1ull, 0x1000f48006010400ull, 0x40000ull, ~0ull}},
          output_array{exec, 4}
    {}

    gko::array<gko::uint32> input_array32;
    gko::array<gko::uint64> input_array64;
    gko::array<int> output_array;
};


// nvcc doesn't like device lambdas declared in complex classes, move it out
template <typename T>
void run_popcount(std::shared_ptr<gko::EXEC_TYPE> exec, gko::size_type size,
                  T* in, int* out)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in, auto out) {
            out[i] = gko::detail::popcount(in[i]);
        },
        size, in, out);
}

TEST_F(Intrinsics, Popcount32)
{
    run_popcount(exec, input_array32.get_size(), input_array32.get_data(),
                 output_array.get_data());

    GKO_ASSERT_ARRAY_EQ(output_array, (I<int>{1, 9, 1, 32}));
}

TEST_F(Intrinsics, Popcount64)
{
    run_popcount(exec, input_array64.get_size(), input_array64.get_data(),
                 output_array.get_data());

    GKO_ASSERT_ARRAY_EQ(output_array, (I<int>{1, 11, 1, 64}));
}


// nvcc doesn't like device lambdas declared in complex classes, move it out
template <typename T>
void run_find_lowest_bit(std::shared_ptr<gko::EXEC_TYPE> exec,
                         gko::size_type size, T* in, int* out)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in, auto out) {
            out[i] = gko::detail::find_lowest_bit(in[i]);
        },
        size, in, out);
}

TEST_F(Intrinsics, FindLowestBit32)
{
    run_find_lowest_bit(exec, input_array32.get_size(),
                        input_array32.get_data(), output_array.get_data());

    GKO_ASSERT_ARRAY_EQ(output_array, (I<int>{0, 0, 8, 0}));
}

TEST_F(Intrinsics, FindLowestBit64)
{
    run_find_lowest_bit(exec, input_array64.get_size(),
                        input_array64.get_data(), output_array.get_data());

    GKO_ASSERT_ARRAY_EQ(output_array, (I<int>{0, 10, 18, 0}));
}


// nvcc doesn't like device lambdas declared in complex classes, move it out
template <typename T>
void run_find_highest_bit(std::shared_ptr<gko::EXEC_TYPE> exec,
                          gko::size_type size, T* in, int* out)
{
    gko::kernels::GKO_DEVICE_NAMESPACE::run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto in, auto out) {
            out[i] = gko::detail::find_highest_bit(in[i]);
        },
        size, in, out);
}

TEST_F(Intrinsics, FindHighestBit32)
{
    run_find_highest_bit(exec, input_array32.get_size(),
                         input_array32.get_data(), output_array.get_data());

    GKO_ASSERT_ARRAY_EQ(output_array, (I<int>{0, 15, 8, 31}));
}

TEST_F(Intrinsics, FindHighestBit64)
{
    run_find_highest_bit(exec, input_array64.get_size(),
                         input_array64.get_data(), output_array.get_data());

    GKO_ASSERT_ARRAY_EQ(output_array, (I<int>{0, 60, 18, 63}));
}
