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


#include "hip/components/cooperative_groups.hip.hpp"


#include <cstring>
#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/test/utils.hpp"
#include "hip/base/types.hip.hpp"


namespace {


using namespace gko::kernels::hip;


class CooperativeGroups : public ::testing::Test {
protected:
    CooperativeGroups()
        : ref(gko::ReferenceExecutor::create()),
          hip(gko::HipExecutor::create(0, ref)),
          result(ref, 1),
          dresult(hip)
    {
        *result.get_data() = true;
        dresult = result;
    }

    template <typename Kernel>
    void test(Kernel kernel)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel), dim3(1),
                           dim3(config::warp_size), 0, 0, dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    template <typename Kernel>
    void test_subwarp(Kernel kernel)
    {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel), dim3(1),
                           dim3(config::warp_size / 2), 0, 0,
                           dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::HipExecutor> hip;
    gko::Array<bool> result;
    gko::Array<bool> dresult;
};


constexpr static int subwarp_size = config::warp_size / 4;


__device__ void test_assert(bool *success, bool partial)
{
    if (!partial) {
        *success = false;
    }
}


__global__ void cg_shuffle(bool *s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto i = int(group.thread_rank());
    test_assert(s, group.shfl_up(i, 1) == max(0, i - 1));
    test_assert(s, group.shfl_down(i, 1) == min(i + 1, config::warp_size - 1));
    test_assert(s, group.shfl(i, 0) == 0);
}


TEST_F(CooperativeGroups, Shuffle) { test(cg_shuffle); }


__global__ void cg_all(bool *s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    test_assert(s, group.all(true));
    test_assert(s, !group.all(false));
    test_assert(s, !group.all(threadIdx.x < 13));
}


TEST_F(CooperativeGroups, All) { test(cg_all); }


__global__ void cg_any(bool *s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    test_assert(s, group.any(true));
    test_assert(s, group.any(threadIdx.x == 0));
    test_assert(s, !group.any(false));
}


TEST_F(CooperativeGroups, Any) { test(cg_any); }


__global__ void cg_ballot(bool *s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    test_assert(s, group.ballot(false) == 0);
    test_assert(s, group.ballot(true) == ~config::lane_mask_type{});
    test_assert(s, group.ballot(threadIdx.x < 4) == 0xf);
}


TEST_F(CooperativeGroups, Ballot) { test(cg_ballot); }


__global__ void cg_subwarp_shuffle(bool *s)
{
    auto group =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto i = int(group.thread_rank());
    test_assert(s, group.shfl_up(i, 1) == max(i - 1, 0));
    test_assert(s, group.shfl_down(i, 1) == min(i + 1, subwarp_size - 1));
    auto group_base = threadIdx.x / subwarp_size * subwarp_size;
    test_assert(s, group.shfl(int(threadIdx.x), 0) == group_base);
    if (threadIdx.x / subwarp_size == 1) {
        test_assert(s, group.shfl_up(i, 1) == max(i - 1, 0));
        test_assert(s, group.shfl_down(i, 1) == min(i + 1, subwarp_size - 1));
        test_assert(s, group.shfl(int(threadIdx.x), 0) == group_base);
    } else {
        test_assert(s, group.shfl_down(i, 1) == min(i + 1, subwarp_size - 1));
        test_assert(s, group.shfl(int(threadIdx.x), 0) == group_base);
        test_assert(s, group.shfl_up(i, 1) == max(i - 1, 0));
    }
}


TEST_F(CooperativeGroups, SubwarpShuffle) { test(cg_subwarp_shuffle); }


TEST_F(CooperativeGroups, SubwarpShuffle2) { test_subwarp(cg_subwarp_shuffle); }


__global__ void cg_subwarp_all(bool *s)
{
    auto grp = threadIdx.x / subwarp_size;
    bool test_grp = grp == 1;
    auto i = threadIdx.x % subwarp_size;
    // only test with test_grp, the other threads run 'interference'
    auto group =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    test_assert(s, !test_grp || group.all(test_grp));
    test_assert(s, !test_grp || !group.all(!test_grp));
    test_assert(s, !test_grp || !group.all(i < subwarp_size - 3 || !test_grp));
    if (test_grp) {
        test_assert(s, group.all(true));
        test_assert(s, !group.all(false));
        test_assert(s, !group.all(i < subwarp_size - 3));
    } else {
        test_assert(s, !group.all(false));
        test_assert(s, !group.all(i < subwarp_size - 3));
        test_assert(s, group.all(true));
    }
}


TEST_F(CooperativeGroups, SubwarpAll) { test(cg_subwarp_all); }


TEST_F(CooperativeGroups, SubwarpAll2) { test_subwarp(cg_subwarp_all); }


__global__ void cg_subwarp_any(bool *s)
{
    auto grp = threadIdx.x / subwarp_size;
    bool test_grp = grp == 1;
    // only test with test_grp, the other threads run 'interference'
    auto group =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto i = group.thread_rank();
    test_assert(s, !test_grp || group.any(test_grp));
    test_assert(s, !test_grp || group.any(test_grp && i == 1));
    test_assert(s, !test_grp || !group.any(!test_grp));
    if (test_grp) {
        test_assert(s, group.any(true));
        test_assert(s, group.any(i == 1));
        test_assert(s, !group.any(false));
    } else {
        test_assert(s, !group.any(false));
        test_assert(s, group.any(true));
        test_assert(s, group.any(i == 1));
    }
}


TEST_F(CooperativeGroups, SubwarpAny) { test(cg_subwarp_any); }


TEST_F(CooperativeGroups, SubwarpAny2) { test_subwarp(cg_subwarp_any); }


__global__ void cg_subwarp_ballot(bool *s)
{
    auto grp = threadIdx.x / subwarp_size;
    bool test_grp = grp == 1;
    auto full_mask = (config::lane_mask_type{1} << subwarp_size) - 1;
    // only test with test_grp, the other threads run 'interference'
    auto group =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto i = group.thread_rank();
    test_assert(s, !test_grp || group.ballot(!test_grp) == 0);
    test_assert(s, !test_grp || group.ballot(test_grp) == full_mask);
    test_assert(s, !test_grp || group.ballot(i < 4 || !test_grp) == 0xf);
    if (test_grp) {
        test_assert(s, group.ballot(false) == 0);
        test_assert(s, group.ballot(true) == full_mask);
        test_assert(s, group.ballot(i < 4) == 0xf);
    } else {
        test_assert(s, group.ballot(true) == full_mask);
        test_assert(s, group.ballot(i < 4) == 0xf);
        test_assert(s, group.ballot(false) == 0);
    }
}


TEST_F(CooperativeGroups, SubwarpBallot) { test(cg_subwarp_ballot); }


TEST_F(CooperativeGroups, SubwarpBallot2) { test_subwarp(cg_subwarp_ballot); }


template <typename ValueType>
__global__ void cg_shuffle_sum(const int num, ValueType *__restrict__ value)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    for (int ind = 0; ind < num; ind++) {
        value[group.thread_rank()] += group.shfl(value[ind], ind);
    }
}


TEST_F(CooperativeGroups, ShuffleSumDouble)
{
    int num = 4;
    uint64_t x = 0x401022C90008B240;
    double x_dbl{};
    std::memcpy(&x_dbl, &x, sizeof(x_dbl));
    gko::Array<double> value(ref, config::warp_size);
    gko::Array<double> answer(ref, config::warp_size);
    gko::Array<double> dvalue(hip);
    for (int i = 0; i < value.get_num_elems(); i++) {
        value.get_data()[i] = x_dbl;
        answer.get_data()[i] = value.get_data()[i] * (1 << num);
    }
    dvalue = value;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(cg_shuffle_sum<double>), dim3(1),
                       dim3(config::warp_size), 0, 0, num, dvalue.get_data());

    value = dvalue;
    GKO_ASSERT_ARRAY_EQ(&value, &answer);
}


TEST_F(CooperativeGroups, ShuffleSumComplexDouble)
{
    int num = 4;
    uint64_t x = 0x401022C90008B240;
    double x_dbl{};
    std::memcpy(&x_dbl, &x, sizeof(x_dbl));
    gko::Array<std::complex<double>> value(ref, config::warp_size);
    gko::Array<std::complex<double>> answer(ref, config::warp_size);
    gko::Array<std::complex<double>> dvalue(hip);
    for (int i = 0; i < value.get_num_elems(); i++) {
        value.get_data()[i] = std::complex<double>{x_dbl, x_dbl};
        answer.get_data()[i] =
            std::complex<double>{x_dbl * (1 << num), x_dbl * (1 << num)};
    }
    dvalue = value;

    hipLaunchKernelGGL(HIP_KERNEL_NAME(cg_shuffle_sum<thrust::complex<double>>),
                       dim3(1), dim3(config::warp_size), 0, 0, num,
                       as_hip_type(dvalue.get_data()));

    value = dvalue;
    GKO_ASSERT_ARRAY_EQ(&value, &answer);
}


}  // namespace
