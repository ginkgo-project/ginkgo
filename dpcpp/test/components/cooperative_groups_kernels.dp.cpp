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

    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::DpcppExecutor> dpcpp;
    gko::Array<bool> result;
    gko::Array<bool> dresult;
};


void test_assert(bool *success, bool partial)
{
    if (!partial) {
        *success = false;
    }
}

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
        cgh.parallel_for(
            sycl_nd_range(grid, block),
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
    auto active = ~(~config::lane_mask_type{} << config::warp_size);
    test_assert(s, group.ballot(false) == 0);
    test_assert(s, group.ballot(true) == (~config::lane_mask_type{} & active));
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


}  // namespace
