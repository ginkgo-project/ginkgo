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

#include "cuda/components/cooperative_groups.cuh"


#include <memory>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


namespace {


using namespace gko::kernels::cuda;


class CooperativeGroups : public ::testing::Test {
protected:
    CooperativeGroups()
        : ref(gko::ReferenceExecutor::create()),
          cuda(gko::CudaExecutor::create(0, ref)),
          result(ref, 1),
          dresult(cuda)
    {
        *result.get_data() = true;
        dresult = result;
    }

    template <typename Kernel>
    void test(Kernel kernel)
    {
        kernel<<<1, config::warp_size>>>(dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::CudaExecutor> cuda;
    gko::Array<bool> result;
    gko::Array<bool> dresult;
};


constexpr static int subwarp_size = config::warp_size / 4;


__device__ void test_assert(bool *success, bool partial)
{
    auto warp =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto full = warp.all(partial);
    if (threadIdx.x == 0) {
        *success &= full;
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
}

TEST_F(CooperativeGroups, SubwarpShuffle) { test(cg_subwarp_shuffle); }


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
}

TEST_F(CooperativeGroups, SubwarpAll) { test(cg_subwarp_all); }


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
}

TEST_F(CooperativeGroups, SubwarpAny) { test(cg_subwarp_any); }


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
}

TEST_F(CooperativeGroups, SubwarpBallot) { test(cg_subwarp_ballot); }


}  // namespace
