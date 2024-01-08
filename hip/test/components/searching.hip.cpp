// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

// force-top: on
// TODO remove when the HIP includes are fixed
#include <hip/hip_runtime.h>
// force-top: off


#include "hip/components/searching.hip.hpp"


#include <memory>
#include <numeric>
#include <vector>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/test/utils.hip.hpp"


namespace {


using namespace gko::kernels::hip;
using namespace gko::kernels::hip::group;


class Searching : public HipTestFixture {
protected:
    Searching() : result(ref, 1), dresult(exec), sizes(14203)
    {
        std::iota(sizes.begin(), sizes.end(), 0);
    }

    template <typename Kernel>
    void run_test(Kernel kernel, int offset, int size, unsigned num_blocks = 1)
    {
        *result.get_data() = true;
        dresult = result;
        kernel<<<num_blocks, config::warp_size, 0, exec->get_stream()>>>(
            dresult.get_data(), offset, size);
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    gko::array<bool> result;
    gko::array<bool> dresult;
    std::vector<int> sizes;
};


__device__ void test_assert(bool* success, bool predicate)
{
    if (!predicate) {
        *success = false;
    }
}


__global__ void test_binary_search(bool* success, int offset, int size)
{
    // test binary search on [offset, offset + size)
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


__global__ void test_empty_binary_search(bool* success, int offset, int)
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


__global__ void test_sync_binary_search(bool* success, int, int size)
{
    // test binary search on [0, size)
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


__global__ void test_empty_sync_binary_search(bool* success, int, int)
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


__global__ void test_warp_ary_search(bool* success, int offset, int size)
{
    // test binary search on [offset, offset + size)
    // for all possible partition points
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
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


__global__ void test_warp_wide_search(bool* success, int offset, int size)
{
    // test binary search on [offset, offset + size)
    // for all possible partition points
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
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
