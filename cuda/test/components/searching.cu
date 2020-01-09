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

#include "cuda/components/searching.cuh"


#include <memory>
#include <numeric>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "cuda/components/cooperative_groups.cuh"


namespace {


using namespace gko::kernels::cuda;
using cooperative_groups::this_thread_block;


class Searching : public ::testing::Test {
protected:
    Searching()
        : ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          result(ref, 1),
          dresult(cuda),
          sizes(14203)
    {
        std::iota(sizes.begin(), sizes.end(), 0);
    }

    template <typename Kernel>
    void run_test(Kernel kernel, int offset, int size, unsigned num_blocks = 1)
    {
        *result.get_data() = true;
        dresult = result;
        kernel<<<num_blocks, config::warp_size>>>(dresult.get_data(), offset,
                                                  size);
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    gko::Array<bool> result;
    gko::Array<bool> dresult;
    std::vector<int> sizes;
};


__device__ void test_assert(bool *success, bool predicate)
{
    if (!predicate) {
        *success = false;
    }
}


__global__ void test_binary_search(bool *success, int offset, int size)
{
    // test binary search on [0, size)
    // for all possible partition points
    auto result = binary_search(offset, size, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, i >= offset && i < offset + size);
        return i >= threadIdx.x + offset;
    });
    auto result2 = binary_search(offset, size, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, i >= offset && i < offset + size);
        return i >= threadIdx.x + offset + 1;
    });
    test_assert(success, result == threadIdx.x + offset);
    test_assert(success, result2 == threadIdx.x + offset + 1);
}

TEST_F(Searching, BinaryNoOffset)
{
    run_test(test_binary_search, 0, config::warp_size);
}

TEST_F(Searching, BinaryOffset)
{
    run_test(test_binary_search, 5, config::warp_size);
}


__global__ void test_empty_binary_search(bool *success, int offset, int)
{
    auto result = binary_search(offset, 0, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, false);
        return false;
    });
    test_assert(success, result == offset);
}

TEST_F(Searching, BinaryEmptyNoOffset)
{
    run_test(test_empty_binary_search, 0, 0);
}

TEST_F(Searching, BinaryEmptyOffset)
{
    run_test(test_empty_binary_search, 5, 0);
}


__global__ void test_sync_binary_search(bool *success, int, int size)
{
    // test binary search on [0, warp_size)
    // for all possible partition points
    auto result = synchronous_binary_search(size, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, i >= 0 && i < size);
        return i >= threadIdx.x;
    });
    auto result2 = synchronous_binary_search(size, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, i >= 0 && i < size);
        return i >= threadIdx.x + 1;
    });
    test_assert(success, result == threadIdx.x);
    test_assert(success, result2 == threadIdx.x + 1);
}

TEST_F(Searching, SyncBinary)
{
    run_test(test_sync_binary_search, 0, config::warp_size);
}


__global__ void test_empty_sync_binary_search(bool *success, int, int)
{
    auto result = synchronous_binary_search(0, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, false);
        return false;
    });
    test_assert(success, result == 0);
}

TEST_F(Searching, EmptySyncBinary)
{
    run_test(test_empty_sync_binary_search, 0, config::warp_size);
}


__global__ void test_warp_ary_search(bool *success, int offset, int size)
{
    // test binary search on [0, length)
    // for all possible partition points
    auto warp = group::tiled_partition<config::warp_size>(this_thread_block());
    auto result = group_ary_search(offset, size, warp, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, i >= offset && i < offset + size);
        return i >= blockIdx.x + offset;
    });
    test_assert(success, result == blockIdx.x + offset);
}

TEST_F(Searching, WarpAryNoOffset)
{
    for (auto size : sizes) {
        run_test(test_warp_ary_search, 0, size, size + 1);
    }
}

TEST_F(Searching, WarpAryOffset)
{
    for (auto size : sizes) {
        run_test(test_warp_ary_search, 134, size, size + 1);
    }
}


__global__ void test_warp_wide_search(bool *success, int offset, int size)
{
    // test binary search on [0, length)
    // for all possible partition points
    auto warp = group::tiled_partition<config::warp_size>(this_thread_block());
    auto result = group_wide_search(offset, size, warp, [&](int i) {
        // don't access out-of-bounds!
        test_assert(success, i >= offset && i < offset + size);
        return i >= blockIdx.x + offset;
    });
    test_assert(success, result == blockIdx.x + offset);
}

TEST_F(Searching, WarpWideNoOffset)
{
    for (auto size : sizes) {
        run_test(test_warp_wide_search, 0, size, size + 1);
    }
}

TEST_F(Searching, WarpWideOffset)
{
    for (auto size : sizes) {
        run_test(test_warp_wide_search, 142, size, size + 1);
    }
}


}  // namespace
