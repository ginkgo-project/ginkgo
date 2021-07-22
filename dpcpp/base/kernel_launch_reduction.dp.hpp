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

#ifndef GKO_COMMON_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/base/kernel_launch_reduction.hpp"
#endif


#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


using KCFG_1D = ConfigSet<11, 7>;
constexpr auto kcfg_1d_list_simple_reduction =
    syn::value_list<int, static_cast<int>(KCFG_1D::encode(512, 64)),
                    static_cast<int>(KCFG_1D::encode(512, 32)),
                    static_cast<int>(KCFG_1D::encode(512, 16)),
                    static_cast<int>(KCFG_1D::encode(256, 32)),
                    static_cast<int>(KCFG_1D::encode(256, 16)),
                    static_cast<int>(KCFG_1D::encode(256, 8))>();


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... KernelArgs>
void generic_kernel_reduction_1d(sycl::handler& cgh, int64 size,
                                 int64 num_workgroups, KernelFunction fn,
                                 ReductionOp op, FinalizeOp finalize,
                                 ValueType init, ValueType* storage,
                                 KernelArgs... args)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    constexpr auto num_partials = wg_size / sg_size;
    sycl::accessor<UninitializedArray<ValueType, num_partials>, 1,
                   sycl::access_mode::read_write, sycl::access::target::local>
        subgroup_partial_acc(sycl::range<1>{1}, cgh);
    const auto range = sycl_nd_range(dim3(num_workgroups), dim3(wg_size));
    const auto global_size = num_workgroups * wg_size;

    cgh.parallel_for(
        range, [=
    ](sycl::nd_item<3> idx) [[intel::reqd_sub_group_size(sg_size)]] {
            auto subgroup_partial = &subgroup_partial_acc[0][0];
            const auto tidx = thread::get_thread_id_flat<int64>(idx);
            const auto local_tidx = static_cast<int64>(tidx % wg_size);
            auto subgroup =
                group::tiled_partition<sg_size>(group::this_thread_block(idx));
            auto partial = init;
            for (int64 i = tidx; i < size; i += global_size) {
                partial = op(partial, fn(i, args...));
            }
            partial = ::gko::kernels::dpcpp::reduce(subgroup, partial, op);
            if (subgroup.thread_rank() == 0) {
                subgroup_partial[local_tidx / sg_size] = partial;
            }
            idx.barrier(sycl::access::fence_space::local_space);
            if (local_tidx < sg_size) {
                partial = init;
                for (int64 i = local_tidx; i < num_partials; i += sg_size) {
                    partial = op(partial, subgroup_partial[i]);
                }
                partial = ::gko::kernels::dpcpp::reduce(subgroup, partial, op);
                if (subgroup.thread_rank() == 0) {
                    storage[tidx / wg_size] = finalize(partial);
                }
            }
        });
}


template <std::uint32_t cfg = KCFG_1D::encode(256, 16), typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... KernelArgs>
void generic_kernel_reduction_2d(sycl::handler& cgh, int64 rows, int64 cols,
                                 int64 num_workgroups, KernelFunction fn,
                                 ReductionOp op, FinalizeOp finalize,
                                 ValueType init, ValueType* storage,
                                 KernelArgs... args)
{
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    constexpr auto num_partials = wg_size / sg_size;
    sycl::accessor<UninitializedArray<ValueType, num_partials>, 1,
                   sycl::access_mode::read_write, sycl::access::target::local>
        subgroup_partial_acc(sycl::range<1>{1}, cgh);
    const auto range = sycl_nd_range(dim3(num_workgroups), dim3(wg_size));
    const auto global_size = num_workgroups * wg_size;

    cgh.parallel_for(
        range, [=
    ](sycl::nd_item<3> idx) [[intel::reqd_sub_group_size(sg_size)]] {
            auto subgroup_partial = &subgroup_partial_acc[0][0];
            const auto tidx = thread::get_thread_id_flat<int64>(idx);
            const auto local_tidx = static_cast<int64>(tidx % wg_size);
            auto subgroup =
                group::tiled_partition<sg_size>(group::this_thread_block(idx));
            auto partial = init;
            for (int64 i = tidx; i < rows * cols; i += global_size) {
                const auto row = i / cols;
                const auto col = i % cols;
                partial = op(partial, fn(row, col, args...));
            }
            partial = ::gko::kernels::dpcpp::reduce(subgroup, partial, op);
            if (subgroup.thread_rank() == 0) {
                subgroup_partial[local_tidx / sg_size] = partial;
            }
            idx.barrier(sycl::access::fence_space::local_space);
            if (local_tidx < sg_size) {
                partial = init;
                for (int64 i = local_tidx; i < num_partials; i += sg_size) {
                    partial = op(partial, subgroup_partial[i]);
                }
                partial = ::gko::kernels::dpcpp::reduce(subgroup, partial, op);
                if (subgroup.thread_rank() == 0) {
                    storage[tidx / wg_size] = finalize(partial);
                }
            }
        });
}


template <int icfg, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction_impl(syn::value_list<int, icfg>,
                               std::shared_ptr<const DpcppExecutor> exec,
                               KernelFunction fn, ReductionOp op,
                               FinalizeOp finalize, ValueType init,
                               ValueType* result, size_type size,
                               KernelArgs... args)
{
    constexpr auto cfg = static_cast<std::uint32_t>(icfg);
    constexpr int oversubscription = 4;
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    const auto num_workgroups =
        std::min<int64>(ceildiv(size, wg_size),
                        exec->get_num_computing_units() * oversubscription);
    auto queue = exec->get_queue();
    if (num_workgroups > 1) {
        Array<ValueType> partial{exec, static_cast<size_type>(num_workgroups)};
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d(
                cgh, static_cast<int64>(size), num_workgroups, fn, op,
                [](auto v) { return v; }, init, partial.get_data(), args...);
        });
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d(
                cgh, static_cast<int64>(num_workgroups), 1,
                [](auto i, auto v) { return v[i]; }, op, finalize, init, result,
                partial.get_const_data());
        });
    } else {
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d(cgh, static_cast<int64>(size),
                                        num_workgroups, fn, op, finalize, init,
                                        result, args...);
        });
    }
}


template <int icfg, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction_impl(syn::value_list<int, icfg>,
                               std::shared_ptr<const DpcppExecutor> exec,
                               KernelFunction fn, ReductionOp op,
                               FinalizeOp finalize, ValueType init,
                               ValueType* result, dim<2> size,
                               KernelArgs... args)
{
    constexpr auto cfg = static_cast<std::uint32_t>(icfg);
    constexpr int oversubscription = 4;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto flat_size = rows * cols;
    constexpr auto wg_size = KCFG_1D::decode<0>(cfg);
    constexpr auto sg_size = KCFG_1D::decode<1>(cfg);
    const auto num_workgroups =
        std::min<int64>(ceildiv(flat_size, wg_size),
                        exec->get_num_computing_units() * oversubscription);
    auto queue = exec->get_queue();
    if (num_workgroups > 1) {
        Array<ValueType> partial{exec, static_cast<size_type>(num_workgroups)};
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_2d(
                cgh, rows, cols, num_workgroups, fn, op,
                [](auto v) { return v; }, init, partial.get_data(), args...);
        });
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d(
                cgh, static_cast<int64>(num_workgroups), 1,
                [](auto i, auto v) { return v[i]; }, op, finalize, init, result,
                partial.get_const_data());
        });
    } else {
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_2d(cgh, rows, cols, num_workgroups, fn, op,
                                        finalize, init, result, args...);
        });
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_reduction,
                                    run_kernel_reduction_impl)


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const DpcppExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType* result, dim<2> size, KernelArgs&&... args)
{
    const auto desired_icfg = static_cast<int>(get_first_cfg(
        as_array(kcfg_1d_list_simple_reduction), [&](std::uint32_t cfg) {
            return validate(exec->get_queue(), KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        }));
    select_run_kernel_reduction(
        kcfg_1d_list_simple_reduction,
        [&](int icfg) { return icfg == desired_icfg; }, syn::value_list<int>(),
        syn::type_list<>(), exec, fn, op, finalize, init, result, size,
        map_to_device(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const DpcppExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType init,
                          ValueType* result, size_type size,
                          KernelArgs&&... args)
{
    const auto desired_icfg = static_cast<int>(get_first_cfg(
        as_array(kcfg_1d_list_simple_reduction), [&](std::uint32_t cfg) {
            return validate(exec->get_queue(), KCFG_1D::decode<0>(cfg),
                            KCFG_1D::decode<1>(cfg));
        }));
    select_run_kernel_reduction(
        kcfg_1d_list_simple_reduction,
        [&](int icfg) { return icfg == desired_icfg; }, syn::value_list<int>(),
        syn::type_list<>(), exec, fn, op, finalize, init, result, size,
        map_to_device(args)...);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
