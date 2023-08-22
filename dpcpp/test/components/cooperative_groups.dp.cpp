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

#include "dpcpp/components/cooperative_groups.dp.hpp"


#include <iostream>
#include <memory>


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/types.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "core/test/utils/assertions.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace {


using namespace gko::kernels::sycl;
constexpr auto default_config_list = dcfg_1sg_list_t();


class CooperativeGroups : public testing::TestWithParam<unsigned int> {
protected:
    CooperativeGroups()
        : ref(gko::ReferenceExecutor::create()),
          sycl(gko::SyclExecutor::create(0, ref)),
          test_case(3),
          max_num(test_case * 64),
          result(ref, max_num),
          dresult(sycl)
    {
        for (int i = 0; i < max_num; i++) {
            result.get_data()[i] = false;
        }
        dresult = result;
    }

    template <typename Kernel>
    void test_all_subgroup(Kernel kernel)
    {
        auto subgroup_size = GetParam();
        auto queue = sycl->get_queue();
        if (gko::kernels::sycl::validate(queue, subgroup_size, subgroup_size)) {
            const auto cfg = DCFG_1D::encode(subgroup_size, subgroup_size);
            for (int i = 0; i < test_case * subgroup_size; i++) {
                result.get_data()[i] = true;
            }

            kernel(cfg, 1, subgroup_size, 0, sycl->get_queue(),
                   dresult.get_data());

            // each subgreoup size segment for one test
            GKO_ASSERT_ARRAY_EQ(result, dresult);
        } else {
            GTEST_SKIP() << "This device does not contain this subgroup size "
                         << subgroup_size;
        }
    }

    int test_case;
    int max_num;
    std::shared_ptr<gko::ReferenceExecutor> ref;
    std::shared_ptr<gko::SyclExecutor> sycl;
    gko::array<bool> result;
    gko::array<bool> dresult;
};


// kernel implementation
template <typename DeviceConfig>
void cg_shuffle(bool* s, ::sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    auto group =
        group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));
    auto i = int(group.thread_rank());

    s[i] = group.shfl_up(i, 1) == sycl::max(0, (int)(i - 1));
    s[i + sg_size] =
        group.shfl_down(i, 1) ==
        sycl::min((unsigned int)(i + 1), (unsigned int)(sg_size - 1));
    s[i + sg_size * 2] = group.shfl(i, 0) == 0;
}

// group all kernel things together
template <typename DeviceConfig>
void cg_shuffle_host(dim3 grid, dim3 block,
                     gko::size_type dynamic_shared_memory, ::sycl::queue* queue,
                     bool* s)
{
    queue->submit([&](::sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](::sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                DeviceConfig::subgroup_size)]] __WG_BOUND__(DeviceConfig::
                                                                block_size) {
                cg_shuffle<DeviceConfig>(s, item_ct1);
            });
    });
}

// config selection
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(cg_shuffle_config,
                                                  cg_shuffle_host, DCFG_1D)

// the call
void cg_shuffle_config_call(std::uint32_t desired_cfg, dim3 grid, dim3 block,
                            gko::size_type dynamic_shared_memory,
                            ::sycl::queue* queue, bool* s)
{
    cg_shuffle_config(
        default_config_list,
        // validate
        [&desired_cfg](std::uint32_t cfg) { return cfg == desired_cfg; },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        grid, block, dynamic_shared_memory, queue, s);
}

TEST_P(CooperativeGroups, Shuffle)
{
    test_all_subgroup(cg_shuffle_config_call);
}


template <typename DeviceConfig>
void cg_all(bool* s, ::sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    auto group =
        group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));
    auto i = int(group.thread_rank());

    s[i] = group.all(true);
    s[i + sg_size] = !group.all(false);
    s[i + sg_size * 2] =
        group.all(item_ct1.get_local_id(2) < 13) == sg_size < 13;
}

GKO_ENABLE_DEFAULT_HOST_CONFIG_TYPE(cg_all, cg_all)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(cg_all, cg_all, DCFG_1D)
GKO_ENABLE_DEFAULT_CONFIG_CALL(cg_all_call, cg_all, default_config_list)

TEST_P(CooperativeGroups, All) { test_all_subgroup(cg_all_call<bool*>); }


template <typename DeviceConfig>
void cg_any(bool* s, ::sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    auto group =
        group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));
    auto i = int(group.thread_rank());

    s[i] = group.any(true);
    s[i + sg_size] = group.any(item_ct1.get_local_id(2) == 0);
    s[i + sg_size * 2] = !group.any(false);
}

GKO_ENABLE_DEFAULT_HOST_CONFIG_TYPE(cg_any, cg_any)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(cg_any, cg_any, DCFG_1D)
GKO_ENABLE_DEFAULT_CONFIG_CALL(cg_any_call, cg_any, default_config_list)

TEST_P(CooperativeGroups, Any) { test_all_subgroup(cg_any_call<bool*>); }


template <typename cfg>
void cg_ballot(bool* s, ::sycl::nd_item<3> item_ct1)
{
    constexpr auto sg_size = cfg::subgroup_size;
    auto group =
        group::tiled_partition<sg_size>(group::this_thread_block(item_ct1));
    auto active = gko::detail::mask<sg_size, config::lane_mask_type>();
    auto i = int(group.thread_rank());

    s[i] = group.ballot(false) == 0;
    s[i + sg_size] = group.ballot(true) == (~config::lane_mask_type{} & active);
    s[i + sg_size * 2] = group.ballot(item_ct1.get_local_id(2) < 4) == 0xf;
}

GKO_ENABLE_DEFAULT_HOST_CONFIG_TYPE(cg_ballot, cg_ballot)
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(cg_ballot, cg_ballot, DCFG_1D)
GKO_ENABLE_DEFAULT_CONFIG_CALL(cg_ballot_call, cg_ballot, default_config_list)

TEST_P(CooperativeGroups, Ballot) { test_all_subgroup(cg_ballot_call<bool*>); }


INSTANTIATE_TEST_SUITE_P(DifferentSubgroup, CooperativeGroups,
                         testing::Values(4, 8, 16, 32, 64),
                         testing::PrintToStringParamName());


}  // namespace
