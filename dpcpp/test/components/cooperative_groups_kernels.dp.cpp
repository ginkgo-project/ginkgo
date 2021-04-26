/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "dpcpp/components/cooperative_groups.dp.hpp"


#include <memory>


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/test/utils.hpp"


namespace {


using namespace gko::kernels::dpcpp;


class CooperativeGroups : public ::testing::Test {
protected:
    CooperativeGroups()
        : ref(gko::ReferenceExecutor::create()),
          dpcpp(gko::DpcppExecutor::create(0, ref)),
          result(ref, 1),
          dresult(dpcpp)
    {
        *result.get_data() = true;
        dresult = result;
    }

    template <typename Kernel>
    void test(Kernel kernel)
    {
        // functioname kernel
        kernel(1, config::warp_size, 0, dpcpp->get_queue(), dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    template <typename Kernel>
    void test_subwarp(Kernel kernel)
    {
        // functioname kernel
        kernel(1, config::warp_size, 0, dpcpp->get_queue(), dresult.get_data());
        result = dresult;
        auto success = *result.get_const_data();

        ASSERT_TRUE(success);
    }

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::DpcppExecutor> dpcpp;
    gko::Array<bool> result;
    gko::Array<bool> dresult;
};


constexpr static int subwarp_size = config::warp_size / 2;


void test_assert(bool *success, bool partial)
{
    if (!partial) {
        *success = false;
    }
}


#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT_AS __attribute__((opencl_constant))
#else
#define CONSTANT_AS
#endif

void cg_shuffle(bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    auto i = int(group.thread_rank());
    test_assert(s, group.shfl_up(i, 1) == sycl::max(0, (int)(i - 1)));
    test_assert(s, group.shfl_down(i, 1) ==
                       sycl::min((unsigned int)(i + 1),
                                 (unsigned int)(config::warp_size - 1)));
    test_assert(s, group.shfl(i, 0) == 0);
}

void cg_shuffle_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_shuffle(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, Shuffle) { test(cg_shuffle_host); }


void cg_all(bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    test_assert(s, group.all(true));
    test_assert(s, !group.all(false));
    test_assert(s, !group.all(item_ct1.get_local_id(2) < 13));
}

void cg_all_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_all(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, All) { test(cg_all_host); }


void cg_any(bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    test_assert(s, group.any(true));
    test_assert(s, group.any(item_ct1.get_local_id(2) == 0));
    test_assert(s, !group.any(false));
}

void cg_any_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_any(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, Any) { test(cg_any_host); }


void cg_ballot(bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    test_assert(s, group.ballot(false) == 0);
    test_assert(s, group.ballot(true) == ~config::lane_mask_type{});
    test_assert(s, group.ballot(item_ct1.get_local_id(2) < 4) == 0xf);
}

void cg_ballot_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                    sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_ballot(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, Ballot) { test(cg_ballot_host); }


void cg_subwarp_shuffle(bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
    auto i = int(group.thread_rank());
    test_assert(s, group.shfl_up(i, 1) == sycl::max((int)(i - 1), 0));
    test_assert(s, group.shfl_down(i, 1) ==
                       sycl::min((int)(i + 1), (int)(subwarp_size - 1)));
    auto group_base = item_ct1.get_local_id(2) / subwarp_size * subwarp_size;
    test_assert(s, group.shfl(int(item_ct1.get_local_id(2)), 0) == group_base);
    if (item_ct1.get_local_id(2) / subwarp_size == 1) {
        test_assert(s, group.shfl_up(i, 1) == sycl::max((int)(i - 1), 0));
        test_assert(s, group.shfl_down(i, 1) ==
                           sycl::min((int)(i + 1), (int)(subwarp_size - 1)));
        test_assert(s,
                    group.shfl(int(item_ct1.get_local_id(2)), 0) == group_base);
    } else {
        test_assert(s, group.shfl_down(i, 1) ==
                           sycl::min((int)(i + 1), (int)(subwarp_size - 1)));
        test_assert(s,
                    group.shfl(int(item_ct1.get_local_id(2)), 0) == group_base);
        test_assert(s, group.shfl_up(i, 1) == sycl::max((int)(i - 1), 0));
    }
}

void cg_subwarp_shuffle_host(dim3 grid, dim3 block,
                             size_t dynamic_shared_memory, sycl::queue *stream,
                             bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(sycl::nd_range<3>(global_range, local_range),
                         [=](sycl::nd_item<3> item_ct1) {
                             cg_subwarp_shuffle(s, item_ct1);
                         });
    });
}

TEST_F(CooperativeGroups, SubwarpShuffle) { test(cg_subwarp_shuffle_host); }

TEST_F(CooperativeGroups, SubwarpShuffle2)
{
    test_subwarp(cg_subwarp_shuffle_host);
}


void cg_subwarp_all(bool *s, sycl::nd_item<3> item_ct1)
{
    auto grp = item_ct1.get_local_id(2) / subwarp_size;
    bool test_grp = grp == 1;
    auto i = item_ct1.get_local_id(2) % subwarp_size;
    // only test with test_grp, the other threads run 'interference'
    auto group = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
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

void cg_subwarp_all_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_subwarp_all(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, SubwarpAll) { test(cg_subwarp_all_host); }

TEST_F(CooperativeGroups, SubwarpAll2) { test_subwarp(cg_subwarp_all_host); }


void cg_subwarp_any(bool *s, sycl::nd_item<3> item_ct1)
{
    auto grp = item_ct1.get_local_id(2) / subwarp_size;
    bool test_grp = grp == 1;
    // only test with test_grp, the other threads run 'interference'
    auto group = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
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

void cg_subwarp_any_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                         sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_subwarp_any(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, SubwarpAny) { test(cg_subwarp_any_host); }

TEST_F(CooperativeGroups, SubwarpAny2) { test_subwarp(cg_subwarp_any_host); }


void cg_subwarp_ballot(bool *s, sycl::nd_item<3> item_ct1)
{
    auto grp = item_ct1.get_local_id(2) / subwarp_size;
    bool test_grp = grp == 1;
    auto full_mask = (config::lane_mask_type{1} << subwarp_size) - 1;
    // only test with test_grp, the other threads run 'interference'
    auto group = group::tiled_partition<subwarp_size>(
        group::this_thread_block(item_ct1));
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

void cg_subwarp_ballot_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                            sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        auto local_range = block.reverse();
        auto global_range = grid.reverse() * local_range;

        cgh.parallel_for(
            sycl::nd_range<3>(global_range, local_range),
            [=](sycl::nd_item<3> item_ct1) { cg_subwarp_ballot(s, item_ct1); });
    });
}

TEST_F(CooperativeGroups, SubwarpBallot) { test(cg_subwarp_ballot_host); }

TEST_F(CooperativeGroups, SubwarpBallot2)
{
    test_subwarp(cg_subwarp_ballot_host);
}


}  // namespace
