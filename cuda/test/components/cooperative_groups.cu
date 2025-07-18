// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "common/cuda_hip/components/cooperative_groups.hpp"

#include <memory>

#include <gtest/gtest.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "cuda/test/utils.hpp"


namespace {


using namespace gko::kernels::cuda;


class CooperativeGroups : public CudaTestFixture {
protected:
    CooperativeGroups() : result(ref, 1), dresult(exec)
    {
        *result.get_data() = true;
        dresult = result;
    }

    template <typename Kernel>
    void test(Kernel kernel)
    {
        kernel<<<1, config::warp_size, 0, exec->get_stream()>>>(
            dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    template <typename Kernel>
    void test_subwarp(Kernel kernel)
    {
        kernel<<<1, config::warp_size / 2, 0, exec->get_stream()>>>(
            dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    gko::array<bool> result;
    gko::array<bool> dresult;
};


constexpr static int subwarp_size = config::warp_size / 4;


__device__ void test_assert(bool* success, bool partial)
{
    if (!partial) {
        *success = false;
    }
}


__global__ void cg_shuffle(bool* s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    auto i = int(group.thread_rank());
    test_assert(s, group.shfl_up(i, 1) == max(0, i - 1));
    test_assert(s, group.shfl_down(i, 1) ==
                       min(i + 1, static_cast<int>(config::warp_size) - 1));
    test_assert(s, group.shfl(i, 0) == 0);
}

TEST_F(CooperativeGroups, Shuffle) { test(cg_shuffle); }


__global__ void cg_all(bool* s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    test_assert(s, group.all(true));
    test_assert(s, !group.all(false));
    test_assert(s, !group.all(threadIdx.x < 13));
}

TEST_F(CooperativeGroups, All) { test(cg_all); }


__global__ void cg_any(bool* s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    test_assert(s, group.any(true));
    test_assert(s, group.any(threadIdx.x == 0));
    test_assert(s, !group.any(false));
}

TEST_F(CooperativeGroups, Any) { test(cg_any); }


__global__ void cg_ballot(bool* s)
{
    auto group =
        group::tiled_partition<config::warp_size>(group::this_thread_block());
    test_assert(s, group::ballot(group, false) == 0);
    test_assert(s, group::ballot(group, true) == ~config::lane_mask_type{});
    test_assert(s, group::ballot(group, threadIdx.x < 4) == 0xf);
}

TEST_F(CooperativeGroups, Ballot) { test(cg_ballot); }


__global__ void cg_subwarp_shuffle(bool* s)
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


__global__ void cg_subwarp_all(bool* s)
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


__global__ void cg_subwarp_any(bool* s)
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


__global__ void cg_subwarp_ballot(bool* s)
{
    auto grp = threadIdx.x / subwarp_size;
    bool test_grp = grp == 1;
    auto full_mask = (config::lane_mask_type{1} << subwarp_size) - 1;
    // only test with test_grp, the other threads run 'interference'
    auto group =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto i = group.thread_rank();
    test_assert(s, !test_grp || group::ballot(group, !test_grp) == 0);
    test_assert(s, !test_grp || group::ballot(group, test_grp) == full_mask);
    test_assert(s,
                !test_grp || group::ballot(group, i < 4 || !test_grp) == 0xf);
    if (test_grp) {
        test_assert(s, group::ballot(group, false) == 0);
        test_assert(s, group::ballot(group, true) == full_mask);
        test_assert(s, group::ballot(group, i < 4) == 0xf);
    } else {
        test_assert(s, group::ballot(group, true) == full_mask);
        test_assert(s, group::ballot(group, i < 4) == 0xf);
        test_assert(s, group::ballot(group, false) == 0);
    }
}

TEST_F(CooperativeGroups, SubwarpBallot) { test(cg_subwarp_ballot); }

TEST_F(CooperativeGroups, SubwarpBallot2) { test_subwarp(cg_subwarp_ballot); }


__global__ void cg_communicator_categorization(bool*)
{
    auto this_block = group::this_thread_block();
    auto tiled_partition =
        group::tiled_partition<config::warp_size>(this_block);
    auto subwarp_partition = group::tiled_partition<subwarp_size>(this_block);

    using not_group = int;
    using this_block_t = decltype(this_block);
    using tiled_partition_t = decltype(tiled_partition);
    using subwarp_partition_t = decltype(subwarp_partition);

    static_assert(!group::is_group<not_group>::value &&
                      group::is_group<this_block_t>::value &&
                      group::is_group<tiled_partition_t>::value &&
                      group::is_group<subwarp_partition_t>::value,
                  "Group check doesn't work.");
    static_assert(
        !group::is_synchronizable_group<not_group>::value &&
            group::is_synchronizable_group<this_block_t>::value &&
            group::is_synchronizable_group<tiled_partition_t>::value &&
            group::is_synchronizable_group<subwarp_partition_t>::value,
        "Synchronizable group check doesn't work.");
    static_assert(!group::is_communicator_group<not_group>::value &&
                      !group::is_communicator_group<this_block_t>::value &&
                      group::is_communicator_group<tiled_partition_t>::value &&
                      group::is_communicator_group<subwarp_partition_t>::value,
                  "Communicator group check doesn't work.");
}

TEST_F(CooperativeGroups, CorrectCategorization)
{
    test(cg_communicator_categorization);
}


}  // namespace
