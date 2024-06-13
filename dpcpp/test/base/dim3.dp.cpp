// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "dpcpp/base/dim3.dp.hpp"


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


namespace {


using namespace gko::kernels::dpcpp;


TEST(DpcppDim3, CanGenerate1DRange)
{
    dim3 block(3);
    auto sycl_block = block.get_range();

    ASSERT_EQ(block.x, 3);
    ASSERT_EQ(block.y, 1);
    ASSERT_EQ(block.z, 1);
    ASSERT_EQ(sycl_block.get(0), 1);
    ASSERT_EQ(sycl_block.get(1), 1);
    ASSERT_EQ(sycl_block.get(2), 3);
}


TEST(DpcppDim3, CanGenerate2DRange)
{
    dim3 block(3, 5);
    auto sycl_block = block.get_range();

    ASSERT_EQ(block.x, 3);
    ASSERT_EQ(block.y, 5);
    ASSERT_EQ(block.z, 1);
    ASSERT_EQ(sycl_block.get(0), 1);
    ASSERT_EQ(sycl_block.get(1), 5);
    ASSERT_EQ(sycl_block.get(2), 3);
}


TEST(DpcppDim3, CanGenerate3DRange)
{
    dim3 block(3, 5, 7);
    auto sycl_block = block.get_range();

    ASSERT_EQ(block.x, 3);
    ASSERT_EQ(block.y, 5);
    ASSERT_EQ(block.z, 7);
    ASSERT_EQ(sycl_block.get(0), 7);
    ASSERT_EQ(sycl_block.get(1), 5);
    ASSERT_EQ(sycl_block.get(2), 3);
}


TEST(DpcppDim3, CanGenerateNDRange)
{
    dim3 block(3, 5, 7);
    dim3 grid(17, 13, 11);

    auto ndrange = sycl_nd_range(grid, block);
    auto global_size = ndrange.get_global_range();
    auto local_size = ndrange.get_local_range();

    ASSERT_EQ(local_size.get(0), 7);
    ASSERT_EQ(local_size.get(1), 5);
    ASSERT_EQ(local_size.get(2), 3);
    ASSERT_EQ(global_size.get(0), 7 * 11);
    ASSERT_EQ(global_size.get(1), 5 * 13);
    ASSERT_EQ(global_size.get(2), 3 * 17);
}


}  // namespace
