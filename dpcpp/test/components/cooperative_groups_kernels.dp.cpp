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


#include <iostream>
#include <memory>


#include <CL/sycl.hpp>


#include <gtest/gtest.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"


namespace {


using namespace gko::kernels::dpcpp;
using KCfg = gko::ConfigSet<12, 7>;
constexpr auto default_config_list =
    ::gko::syn::value_list<int, KCfg::encode(64, 64), KCfg::encode(32, 32),
                           KCfg::encode(16, 16), KCfg::encode(8, 8),
                           KCfg::encode(4, 4)>();

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
    void test_all_subgroup(Kernel kernel)
    {
        auto exec_info = dpcpp->get_const_exec_info();
        for (auto &i : exec_info.subgroup_sizes) {
            kernel(1, i, 0, dpcpp->get_queue(), dpcpp, dresult.get_data());
            result = dresult;
            auto success = *result.get_const_data();
            ASSERT_TRUE(success);
            std::cout << i << " success" << std::endl;
        }
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

// kernel implementation
template <int config>
[[intel::reqd_work_group_size(1, 1, KCfg::decode<0>(config))]] void cg_shuffle(
    bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<KCfg::decode<1>(config)>(
        group::this_thread_block(item_ct1));
    auto i = int(group.thread_rank());
    test_assert(s, group.shfl_up(i, 1) == sycl::max(0, (int)(i - 1)));
    test_assert(s, group.shfl_down(i, 1) ==
                       sycl::min((unsigned int)(i + 1),
                                 (unsigned int)(KCfg::decode<1>(config) - 1)));
    test_assert(s, group.shfl(i, 0) == 0);
}

// group all kernel things together
template <int config>
void cg_shuffle_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                     sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             cg_shuffle<config>(s, item_ct1);
                         });
    });
}

// config selection
GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(cg_shuffle_config, cg_shuffle_host)

// the call
void cg_shuffle_config_call(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                            sycl::queue *stream,
                            std::shared_ptr<const gko::DpcppExecutor> exec,
                            bool *s)
{
    auto exec_info = exec->get_const_exec_info();
    cg_shuffle_config(
        default_config_list,
        // validate
        [&exec_info, &block](int config) {
            return exec_info.validate(KCfg::decode<0>(config),
                                      KCfg::decode<1>(config)) &&
                   (KCfg::decode<0>(config) == block.x);
        },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        grid, block, dynamic_shared_memory, stream, s);
}

TEST_F(CooperativeGroups, Shuffle)
{
    test_all_subgroup(cg_shuffle_config_call);
}


template <int config>
[[intel::reqd_work_group_size(1, 1, KCfg::decode<0>(config))]] void cg_all(
    bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<KCfg::decode<1>(config)>(
        group::this_thread_block(item_ct1));
    test_assert(s, group.all(true));
    test_assert(s, !group.all(false));
    test_assert(s, group.all(item_ct1.get_local_id(2) < 13) ==
                       KCfg::decode<1>(config) < 13);
}

template <int config>
void cg_all_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) { cg_all<config>(s, item_ct1); });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(cg_all_config, cg_all_host)

void cg_all_config_call(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                        sycl::queue *stream,
                        std::shared_ptr<const gko::DpcppExecutor> exec, bool *s)
{
    auto exec_info = exec->get_const_exec_info();
    cg_all_config(
        default_config_list,
        // validate
        [&exec_info, &block](int config) {
            return exec_info.validate(KCfg::decode<0>(config),
                                      KCfg::decode<1>(config)) &&
                   (KCfg::decode<0>(config) == block.x);
        },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        grid, block, dynamic_shared_memory, stream, s);
}

TEST_F(CooperativeGroups, All) { test_all_subgroup(cg_all_config_call); }


template <int config>
[[intel::reqd_work_group_size(1, 1, KCfg::decode<0>(config))]] void cg_any(
    bool *s, sycl::nd_item<3> item_ct1)
{
    auto group = group::tiled_partition<KCfg::decode<1>(config)>(
        group::this_thread_block(item_ct1));
    test_assert(s, group.any(true));
    test_assert(s, group.any(item_ct1.get_local_id(2) == 0));
    test_assert(s, !group.any(false));
}

template <int config>
void cg_any_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                 sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) { cg_any<config>(s, item_ct1); });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(cg_any_config, cg_any_host)

void cg_any_config_call(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                        sycl::queue *stream,
                        std::shared_ptr<const gko::DpcppExecutor> exec, bool *s)
{
    auto exec_info = exec->get_const_exec_info();
    cg_any_config(
        default_config_list,
        // validate
        [&exec_info, &block](int config) {
            return exec_info.validate(KCfg::decode<0>(config),
                                      KCfg::decode<1>(config)) &&
                   (KCfg::decode<0>(config) == block.x);
        },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        grid, block, dynamic_shared_memory, stream, s);
}

TEST_F(CooperativeGroups, Any) { test_all_subgroup(cg_any_config_call); }


template <int Size>
config::lane_mask_type active_mask()
{
    return (config::lane_mask_type{1} << Size) - 1;
}

template <>
config::lane_mask_type active_mask<64>()
{
    return ~config::lane_mask_type{};
}


template <int config>
[[intel::reqd_work_group_size(1, 1, KCfg::decode<0>(config))]] void cg_ballot(
    bool *s, sycl::nd_item<3> item_ct1)
{
    constexpr auto subgroup_size = KCfg::decode<1>(config);
    auto group = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));
    auto active = active_mask<subgroup_size>();
    test_assert(s, group.ballot(false) == 0);
    test_assert(s, group.ballot(true) == (~config::lane_mask_type{} & active));
    test_assert(s, group.ballot(item_ct1.get_local_id(2) < 4) == 0xf);
}

template <int config>
void cg_ballot_host(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                    sycl::queue *stream, bool *s)
{
    stream->submit([&](sycl::handler &cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1) { cg_ballot<config>(s, item_ct1); });
    });
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION(cg_ballot_config, cg_ballot_host)

void cg_ballot_config_call(dim3 grid, dim3 block, size_t dynamic_shared_memory,
                           sycl::queue *stream,
                           std::shared_ptr<const gko::DpcppExecutor> exec,
                           bool *s)
{
    auto exec_info = exec->get_const_exec_info();
    cg_ballot_config(
        default_config_list,
        // validate
        [&exec_info, &block](int config) {
            return exec_info.validate(KCfg::decode<0>(config),
                                      KCfg::decode<1>(config)) &&
                   (KCfg::decode<0>(config) == block.x);
        },
        ::gko::syn::value_list<bool>(), ::gko::syn::value_list<int>(),
        ::gko::syn::value_list<gko::size_type>(), ::gko::syn::type_list<>(),
        grid, block, dynamic_shared_memory, stream, s);
}

TEST_F(CooperativeGroups, Ballot) { test_all_subgroup(cg_ballot_config_call); }


}  // namespace
