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

// TODO remove when the HIP includes are fixed
#include <hip/hip_runtime.h>


#include "hip/components/merging.hip.hpp"


#include <algorithm>
#include <memory>
#include <random>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "hip/components/cooperative_groups.hip.hpp"


namespace {


using namespace gko::kernels::hip;
using namespace gko::kernels::hip::group;


class Merging : public ::testing::Test {
protected:
    Merging()
        : ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref)),
          rng(123456),
          rng_runs{100},
          rng_run{},
          max_size{1637},
          sizes{0,  1,  2,   3,   4,   10,  15,   16,
                31, 34, 102, 242, 534, 956, 1239, 1637},
          data1(ref, max_size),
          data2(ref, max_size),
          outdata(ref, 2 * max_size),
          idxs1(ref),
          idxs2(ref),
          idxs3(ref),
          refidxs1(ref),
          refidxs2(ref),
          refidxs3(ref),
          refdata(ref, 2 * max_size),
          ddata1(hip),
          ddata2(hip),
          didxs1(hip, 2 * max_size),
          didxs2(hip, 2 * max_size),
          didxs3(hip, 2 * max_size),
          drefidxs1(hip, 2 * max_size),
          drefidxs2(hip, 2 * max_size),
          drefidxs3(hip, 2 * max_size),
          doutdata(hip, 2 * max_size)
    {}

    void init_data()
    {
        std::uniform_int_distribution<gko::int32> dist(0, max_size);
        std::fill_n(data1.get_data(), max_size, 0);
        std::fill_n(data2.get_data(), max_size, 0);
        for (auto i = 0; i < max_size; ++i) {
            // here we also want to test some corner cases
            // first two runs: zero data1
            if (rng_run > 0) data1.get_data()[i] = dist(rng);
            // first and third run: zero data2
            if (rng_run > 3 || rng_run == 1) data2.get_data()[i] = dist(rng);
        }
        std::sort(data1.get_data(), data1.get_data() + max_size);
        std::sort(data2.get_data(), data2.get_data() + max_size);

        ddata1 = data1;
        ddata2 = data2;
    }

    void assert_eq_ref(int size, int eq_size)
    {
        outdata = doutdata;
        auto out_ptr = outdata.get_const_data();
        auto out_end = out_ptr + eq_size;
        auto ref_ptr = refdata.get_data();
        std::copy_n(data1.get_const_data(), size, ref_ptr);
        std::copy_n(data2.get_const_data(), size, ref_ptr + size);
        std::sort(ref_ptr, ref_ptr + 2 * size);

        ASSERT_TRUE(std::equal(out_ptr, out_end, ref_ptr));
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
    std::default_random_engine rng;

    int rng_runs;
    int rng_run;
    int max_size;
    std::vector<int> sizes;
    gko::Array<gko::int32> data1;
    gko::Array<gko::int32> data2;
    gko::Array<gko::int32> idxs1;
    gko::Array<gko::int32> idxs2;
    gko::Array<gko::int32> idxs3;
    gko::Array<gko::int32> refidxs1;
    gko::Array<gko::int32> refidxs2;
    gko::Array<gko::int32> refidxs3;
    gko::Array<gko::int32> outdata;
    gko::Array<gko::int32> refdata;
    gko::Array<gko::int32> ddata1;
    gko::Array<gko::int32> ddata2;
    gko::Array<gko::int32> didxs1;
    gko::Array<gko::int32> didxs2;
    gko::Array<gko::int32> didxs3;
    gko::Array<gko::int32> drefidxs1;
    gko::Array<gko::int32> drefidxs2;
    gko::Array<gko::int32> drefidxs3;
    gko::Array<gko::int32> doutdata;
};


__global__ void test_merge_step(const gko::int32 *a, const gko::int32 *b,
                                gko::int32 *c)
{
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
    auto i = warp.thread_rank();
    auto result = group_merge_step<config::warp_size>(a[i], b[i], warp);
    c[i] = min(result.a_val, result.b_val);
}

TEST_F(Merging, MergeStep)
{
    for (auto i = 0; i < rng_runs; ++i) {
        init_data();
        hipLaunchKernelGGL(HIP_KERNEL_NAME(test_merge_step), dim3(1),
                           dim3(config::warp_size), 0, 0,
                           ddata1.get_const_data(), ddata2.get_const_data(),
                           doutdata.get_data());

        assert_eq_ref(config::warp_size, config::warp_size);
    }
}


__global__ void test_merge(const gko::int32 *a, const gko::int32 *b, int size,
                           gko::int32 *c)
{
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
    group_merge<config::warp_size>(
        a, size, b, size, warp,
        [&](int a_idx, gko::int32 a_val, int b_idx, gko::int32 b_val, int i) {
            c[i] = min(a_val, b_val);
        });
}

TEST_F(Merging, FullMerge)
{
    for (auto i = 0; i < rng_runs; ++i) {
        init_data();
        for (auto size : sizes) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(test_merge), dim3(1),
                               dim3(config::warp_size), 0, 0,
                               ddata1.get_const_data(), ddata2.get_const_data(),
                               size, doutdata.get_data());

            assert_eq_ref(size, 2 * size);
        }
    }
}


__global__ void test_sequential_merge(const gko::int32 *a, const gko::int32 *b,
                                      int size, gko::int32 *c)
{
    sequential_merge(
        a, size, b, size,
        [&](int a_idx, gko::int32 a_val, int b_idx, gko::int32 b_val, int i) {
            c[i] = min(a_val, b_val);
        });
}

TEST_F(Merging, SequentialFullMerge)
{
    for (auto i = 0; i < rng_runs; ++i) {
        init_data();
        for (auto size : sizes) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(test_sequential_merge), dim3(1),
                               dim3(1), 0, 0, ddata1.get_const_data(),
                               ddata2.get_const_data(), size,
                               doutdata.get_data());

            assert_eq_ref(size, 2 * size);
        }
    }
}


__global__ void test_merge_idxs(const gko::int32 *a, const gko::int32 *b,
                                int size, gko::int32 *c, gko::int32 *aidxs,
                                gko::int32 *bidxs, gko::int32 *cidxs,
                                gko::int32 *refaidxs, gko::int32 *refbidxs,
                                gko::int32 *refcidxs)
{
    if (threadIdx.x == 0) {
        sequential_merge(a, size, b, size,
                         [&](int a_idx, gko::int32 a_val, int b_idx,
                             gko::int32 b_val, int i) {
                             refaidxs[i] = a_idx;
                             refbidxs[i] = b_idx;
                             refcidxs[i] = i;
                         });
    }
    auto warp = tiled_partition<config::warp_size>(this_thread_block());
    group_merge<config::warp_size>(
        a, size, b, size, warp,
        [&](int a_idx, gko::int32 a_val, int b_idx, gko::int32 b_val, int i) {
            aidxs[i] = a_idx;
            bidxs[i] = b_idx;
            cidxs[i] = i;
            c[i] = min(a_val, b_val);
        });
}

TEST_F(Merging, FullMergeIdxs)
{
    for (auto i = 0; i < rng_runs; ++i) {
        init_data();
        for (auto size : sizes) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(test_merge_idxs), dim3(1),
                               dim3(config::warp_size), 0, 0,
                               ddata1.get_const_data(), ddata2.get_const_data(),
                               size, doutdata.get_data(), didxs1.get_data(),
                               didxs2.get_data(), didxs3.get_data(),
                               drefidxs1.get_data(), drefidxs2.get_data(),
                               drefidxs3.get_data());

            assert_eq_ref(size, 2 * size);
            idxs1 = didxs1;
            idxs2 = didxs2;
            idxs3 = didxs3;
            refidxs1 = drefidxs1;
            refidxs2 = drefidxs2;
            refidxs3 = drefidxs3;
            auto idxs1_ptr = idxs1.get_const_data();
            auto idxs2_ptr = idxs2.get_const_data();
            auto idxs3_ptr = idxs3.get_const_data();
            auto refidxs1_ptr = refidxs1.get_const_data();
            auto refidxs2_ptr = refidxs2.get_const_data();
            auto refidxs3_ptr = refidxs3.get_const_data();

            ASSERT_TRUE(
                std::equal(idxs1_ptr, idxs1_ptr + 2 * size, refidxs1_ptr));
            ASSERT_TRUE(
                std::equal(idxs2_ptr, idxs2_ptr + 2 * size, refidxs2_ptr));
            ASSERT_TRUE(
                std::equal(idxs3_ptr, idxs3_ptr + 2 * size, refidxs3_ptr));
        }
    }
}


}  // namespace
