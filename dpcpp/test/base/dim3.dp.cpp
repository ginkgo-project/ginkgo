/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
