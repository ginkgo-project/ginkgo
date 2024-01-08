// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/base/kernel_launch_reduction.hpp"
#endif


#include <algorithm>


#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"
#include "dpcpp/components/uninitialized_array.hpp"
#include "dpcpp/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


static constexpr auto dcfg_1d_list_simple_reduction = dcfg_1d_list_t();


template <typename DeviceConfig, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void generic_kernel_reduction_1d(sycl::handler& cgh, int64 size,
                                 int64 num_workgroups, KernelFunction fn,
                                 ReductionOp op, FinalizeOp finalize,
                                 ValueType identity, ValueType* storage,
                                 MappedKernelArgs... args)
{
    constexpr auto wg_size = DeviceConfig::block_size;
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    constexpr auto num_partials = wg_size / sg_size;
    sycl::accessor<uninitialized_array<ValueType, num_partials>, 0,
                   sycl::access_mode::read_write, sycl::access::target::local>
        subgroup_partial_acc(cgh);
    const auto range = sycl_nd_range(dim3(num_workgroups), dim3(wg_size));
    const auto global_size = num_workgroups * wg_size;

    cgh.parallel_for(
        range,
        [=](sycl::nd_item<3> idx) [[sycl::reqd_sub_group_size(sg_size)]] {
            auto subgroup_partial = &(*subgroup_partial_acc.get_pointer())[0];
            const auto tidx = thread::get_thread_id_flat<int64>(idx);
            const auto local_tidx = static_cast<int64>(tidx % wg_size);
            auto subgroup =
                group::tiled_partition<sg_size>(group::this_thread_block(idx));
            auto partial = identity;
            for (int64 i = tidx; i < size; i += global_size) {
                partial = op(partial, fn(i, args...));
            }
            partial = ::gko::kernels::dpcpp::reduce(subgroup, partial, op);
            if (subgroup.thread_rank() == 0) {
                subgroup_partial[local_tidx / sg_size] = partial;
            }
            idx.barrier(sycl::access::fence_space::local_space);
            if (local_tidx < sg_size) {
                partial = identity;
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


template <typename DeviceConfig, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void generic_kernel_reduction_2d(sycl::handler& cgh, int64 rows, int64 cols,
                                 int64 num_workgroups, KernelFunction fn,
                                 ReductionOp op, FinalizeOp finalize,
                                 ValueType identity, ValueType* storage,
                                 MappedKernelArgs... args)
{
    constexpr auto wg_size = DeviceConfig::block_size;
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    constexpr auto num_partials = wg_size / sg_size;
    sycl::accessor<uninitialized_array<ValueType, num_partials>, 0,
                   sycl::access_mode::read_write, sycl::access::target::local>
        subgroup_partial_acc(cgh);
    const auto range = sycl_nd_range(dim3(num_workgroups), dim3(wg_size));
    const auto global_size = num_workgroups * wg_size;

    cgh.parallel_for(
        range,
        [=](sycl::nd_item<3> idx) [[sycl::reqd_sub_group_size(sg_size)]] {
            auto subgroup_partial = &(*subgroup_partial_acc.get_pointer())[0];
            const auto tidx = thread::get_thread_id_flat<int64>(idx);
            const auto local_tidx = static_cast<int64>(tidx % wg_size);
            auto subgroup =
                group::tiled_partition<sg_size>(group::this_thread_block(idx));
            auto partial = identity;
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
                partial = identity;
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


template <typename DeviceConfig, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_reduction_impl(std::shared_ptr<const DpcppExecutor> exec,
                               KernelFunction fn, ReductionOp op,
                               FinalizeOp finalize, ValueType identity,
                               ValueType* result, size_type size,
                               array<char>& tmp, MappedKernelArgs... args)
{
    constexpr int oversubscription = 4;
    constexpr auto wg_size = DeviceConfig::block_size;
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    const auto num_workgroups =
        std::min<int64>(ceildiv(size, wg_size),
                        exec->get_num_computing_units() * oversubscription);
    auto queue = exec->get_queue();
    if (num_workgroups > 1) {
        const auto required_storage = sizeof(ValueType) * num_workgroups;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d<DeviceConfig>(
                cgh, static_cast<int64>(size), num_workgroups, fn, op,
                [](auto v) { return v; }, identity,
                reinterpret_cast<ValueType*>(tmp.get_data()), args...);
        });
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d<DeviceConfig>(
                cgh, static_cast<int64>(num_workgroups), 1,
                [](auto i, auto v) { return v[i]; }, op, finalize, identity,
                result,
                reinterpret_cast<const ValueType*>(tmp.get_const_data()));
        });
    } else {
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d<DeviceConfig>(
                cgh, static_cast<int64>(size), 1, fn, op, finalize, identity,
                result, args...);
        });
    }
}


template <typename DeviceConfig, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_reduction_impl(std::shared_ptr<const DpcppExecutor> exec,
                               KernelFunction fn, ReductionOp op,
                               FinalizeOp finalize, ValueType identity,
                               ValueType* result, dim<2> size, array<char>& tmp,
                               MappedKernelArgs... args)
{
    constexpr int oversubscription = 4;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto flat_size = rows * cols;
    constexpr auto wg_size = DeviceConfig::block_size;
    constexpr auto sg_size = DeviceConfig::subgroup_size;
    const auto num_workgroups =
        std::min<int64>(ceildiv(flat_size, wg_size),
                        exec->get_num_computing_units() * oversubscription);
    auto queue = exec->get_queue();
    if (num_workgroups > 1) {
        const auto required_storage = sizeof(ValueType) * num_workgroups;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_2d<DeviceConfig>(
                cgh, rows, cols, num_workgroups, fn, op,
                [](auto v) { return v; }, identity,
                reinterpret_cast<ValueType*>(tmp.get_data()), args...);
        });
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_1d<DeviceConfig>(
                cgh, static_cast<int64>(num_workgroups), 1,
                [](auto i, auto v) { return v[i]; }, op, finalize, identity,
                result,
                reinterpret_cast<const ValueType*>(tmp.get_const_data()));
        });
    } else {
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_2d<DeviceConfig>(cgh, rows, cols, 1, fn,
                                                      op, finalize, identity,
                                                      result, args...);
        });
    }
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(select_run_kernel_reduction,
                                                  run_kernel_reduction_impl,
                                                  DCFG_1D)


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction_cached(std::shared_ptr<const DpcppExecutor> exec,
                                 KernelFunction fn, ReductionOp op,
                                 FinalizeOp finalize, ValueType identity,
                                 ValueType* result, dim<2> size,
                                 array<char>& tmp, KernelArgs&&... args)
{
    const auto desired_cfg = get_first_cfg(
        as_array(dcfg_1d_list_simple_reduction), [&](std::uint32_t cfg) {
            return validate(exec->get_queue(), DCFG_1D::decode<0>(cfg),
                            DCFG_1D::decode<1>(cfg));
        });
    select_run_kernel_reduction(
        dcfg_1d_list_simple_reduction,
        [&](std::uint32_t cfg) { return cfg == desired_cfg; },
        syn::value_list<bool>(), syn::value_list<int>(),
        syn::value_list<size_type>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, size, tmp, map_to_device(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction_cached(std::shared_ptr<const DpcppExecutor> exec,
                                 KernelFunction fn, ReductionOp op,
                                 FinalizeOp finalize, ValueType identity,
                                 ValueType* result, size_type size,
                                 array<char>& tmp, KernelArgs&&... args)
{
    const auto desired_cfg = get_first_cfg(
        as_array(dcfg_1d_list_simple_reduction), [&](std::uint32_t cfg) {
            return validate(exec->get_queue(), DCFG_1D::decode<0>(cfg),
                            DCFG_1D::decode<1>(cfg));
        });
    select_run_kernel_reduction(
        dcfg_1d_list_simple_reduction,
        [&](std::uint32_t cfg) { return cfg == desired_cfg; },
        syn::value_list<bool>(), syn::value_list<int>(),
        syn::value_list<size_type>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, size, tmp, map_to_device(args)...);
}


namespace {


template <typename cfg, int ssg_size, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void generic_kernel_row_reduction_2d(syn::value_list<int, ssg_size>,
                                     std::shared_ptr<const DpcppExecutor> exec,
                                     int64 rows, int64 cols, int64 col_blocks,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, int64 result_stride,
                                     MappedKernelArgs... args)
{
    constexpr auto wg_size = cfg::block_size;
    constexpr auto sg_size = cfg::subgroup_size;
    static_assert(ssg_size <= sg_size, "ssg must be smaller than sg");
    const auto num_workgroups = ceildiv(rows * col_blocks * ssg_size, wg_size);
    const auto range = sycl_nd_range(dim3(num_workgroups), dim3(wg_size));
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            range,
            [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(sg_size)]] {
                const auto idx =
                    thread::get_subwarp_id_flat<ssg_size, int64>(id);
                const auto row = idx % rows;
                const auto col_block = idx / rows;
                auto partial = identity;
                auto subgroup = group::tiled_partition<sg_size>(
                    group::this_thread_block(id));
                auto ssg_rank =
                    static_cast<int64>(subgroup.thread_rank() % ssg_size);
                if (col_block < col_blocks) {
                    const auto cols_per_part =
                        ceildiv(ceildiv(cols, ssg_size), col_blocks) * ssg_size;
                    const auto begin = cols_per_part * col_block;
                    const auto end = min(begin + cols_per_part, cols);
                    for (auto col = begin + ssg_rank; col < end;
                         col += ssg_size) {
                        partial = op(partial, fn(row, col, args...));
                    }
                }
// since we do a sub-subgroup reduction, we can't use reduce
#pragma unroll
                for (int i = 1; i < ssg_size; i *= 2) {
                    partial = op(partial, subgroup.shfl_xor(partial, i));
                }
                if (col_block < col_blocks && ssg_rank == 0) {
                    result[(row + col_block * rows) * result_stride] =
                        finalize(partial);
                }
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generic_kernel_row_reduction_2d,
                                    generic_kernel_row_reduction_2d);


template <typename cfg, int ssg_size, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void generic_kernel_col_reduction_2d_small(
    sycl::handler& cgh, int64 rows, int64 cols, int64 row_blocks,
    KernelFunction fn, ReductionOp op, FinalizeOp finalize, ValueType identity,
    ValueType* result, MappedKernelArgs... args)
{
    constexpr auto wg_size = cfg::block_size;
    constexpr auto sg_size = cfg::subgroup_size;
    static_assert(ssg_size <= sg_size, "ssg must be smaller than sg");
    constexpr auto subgroups_per_workgroup = wg_size / sg_size;
    // stores the subwarp_size partial sums from each warp, grouped by warp
    constexpr auto shared_storage = subgroups_per_workgroup * ssg_size;
    sycl::accessor<uninitialized_array<ValueType, shared_storage>, 0,
                   sycl::access_mode::read_write, sycl::access::target::local>
        block_partial_acc(cgh);
    const auto range = sycl_nd_range(dim3(row_blocks), dim3(wg_size));
    cgh.parallel_for(
        range, [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(sg_size)]] {
            auto block_partial = &(*block_partial_acc.get_pointer())[0];
            const auto ssg_id =
                thread::get_subwarp_id_flat<ssg_size, int64>(id);
            const auto local_sg_id = id.get_local_id(2) / sg_size;
            const auto local_ssg_id = id.get_local_id(2) % sg_size / ssg_size;
            const auto ssg_num =
                thread::get_subwarp_num_flat<ssg_size, int64>(id);
            const auto workgroup = group::this_thread_block(id);
            const auto subgroup = group::tiled_partition<sg_size>(workgroup);
            const auto sg_rank = subgroup.thread_rank();
            const auto ssg_rank = sg_rank % ssg_size;
            const auto col = static_cast<int64>(ssg_rank);
            auto partial = identity;
            // accumulate within a thread
            if (col < cols) {
                for (auto row = ssg_id; row < rows; row += ssg_num) {
                    partial = op(partial, fn(row, col, args...));
                }
            }
        // accumulate between all subsubgroups in the subgroup
#pragma unroll
            for (unsigned i = ssg_size; i < sg_size; i *= 2) {
                partial = op(partial, subgroup.shfl_xor(partial, i));
            }
            // store the result to shared memory
            if (local_ssg_id == 0) {
                block_partial[local_sg_id * ssg_size + ssg_rank] = partial;
            }
            workgroup.sync();
            // in a single thread: accumulate the results
            if (local_sg_id == 0) {
                partial = identity;
                // accumulate the partial results within a thread
                if (shared_storage >= sg_size) {
#pragma unroll
                    for (int i = 0; i < shared_storage; i += sg_size) {
                        partial = op(partial, block_partial[i + sg_rank]);
                    }
                } else if (sg_rank < shared_storage) {
                    partial = op(partial, block_partial[sg_rank]);
                }
            // accumulate between all subsubgroups in the subgroup
#pragma unroll
                for (unsigned i = ssg_size; i < sg_size; i *= 2) {
                    partial = op(partial, subgroup.shfl_xor(partial, i));
                }
                if (sg_rank < cols) {
                    result[sg_rank + id.get_group(2) * cols] =
                        finalize(partial);
                }
            }
        });
}


template <typename cfg, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void generic_kernel_col_reduction_2d_blocked(
    sycl::handler& cgh, int64 rows, int64 cols, int64 row_blocks,
    int64 col_blocks, KernelFunction fn, ReductionOp op, FinalizeOp finalize,
    ValueType identity, ValueType* result, MappedKernelArgs... args)
{
    constexpr auto wg_size = cfg::block_size;
    constexpr auto sg_size = cfg::subgroup_size;
    const auto range =
        sycl_nd_range(dim3(row_blocks, col_blocks), dim3(wg_size));
    sycl::accessor<uninitialized_array<ValueType, wg_size>, 0,
                   sycl::access_mode::read_write, sycl::access::target::local>
        block_partial_acc(cgh);
    cgh.parallel_for(
        range, [=](sycl::nd_item<3> id) [[sycl::reqd_sub_group_size(sg_size)]] {
            const auto sg_id = thread::get_subwarp_id_flat<sg_size, int64>(id);
            const auto sg_num =
                thread::get_subwarp_num_flat<sg_size, int64>(id);
            const auto workgroup = group::this_thread_block(id);
            const auto subgroup = group::tiled_partition<sg_size>(workgroup);
            const auto sg_rank = subgroup.thread_rank();
            const auto col =
                sg_rank + static_cast<int64>(id.get_group(1)) * sg_size;
            auto block_partial = &(*block_partial_acc.get_pointer())[0];
            auto partial = identity;
            // accumulate within a thread
            if (col < cols) {
                for (auto row = sg_id; row < rows; row += sg_num) {
                    partial = op(partial, fn(row, col, args...));
                }
            }
            block_partial[id.get_local_id(2)] = partial;
            workgroup.sync();
            // in a single warp: accumulate the results
            if (id.get_local_id(2) < sg_size) {
                partial = identity;
            // accumulate the partial results within a thread
#pragma unroll
                for (int i = 0; i < wg_size; i += sg_size) {
                    partial = op(partial, block_partial[i + sg_rank]);
                }
                if (col < cols) {
                    result[col + id.get_group(2) * cols] = finalize(partial);
                }
            }
        });
}


template <typename ValueType, typename ReductionOp, typename FinalizeOp>
void generic_kernel_reduction_finalize_2d(
    sycl::handler& cgh, int64 num_results, int64 num_blocks, ReductionOp op,
    FinalizeOp finalize, ValueType identity, const ValueType* input,
    int64 result_stride, ValueType* result)
{
    cgh.parallel_for(sycl::range<1>{static_cast<std::size_t>(num_results)},
                     [=](sycl::id<1> id) {
                         auto partial = identity;
                         for (int64 block = 0; block < num_blocks; block++) {
                             partial = op(partial,
                                          input[id[0] + block * num_results]);
                         }
                         result[id[0] * result_stride] = finalize(partial);
                     });
}


template <typename cfg, int ssg_size, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_generic_col_reduction_small(syn::value_list<int, ssg_size>,
                                     std::shared_ptr<const DpcppExecutor> exec,
                                     int64 max_workgroups, KernelFunction fn,
                                     ReductionOp op, FinalizeOp finalize,
                                     ValueType identity, ValueType* result,
                                     dim<2> size, array<char>& tmp,
                                     MappedKernelArgs... args)
{
    constexpr auto wg_size = cfg::block_size;
    constexpr auto sg_size = cfg::subgroup_size;
    static_assert(ssg_size <= sg_size, "ssg must be smaller than sg");
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto row_blocks =
        std::min<int64>(ceildiv(rows * ssg_size, wg_size), max_workgroups);
    auto queue = exec->get_queue();
    if (row_blocks <= 1) {
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_col_reduction_2d_small<cfg, ssg_size>(
                cgh, rows, cols, 1, fn, op, finalize, identity, result,
                args...);
        });
    } else {
        const auto required_storage = sizeof(ValueType) * row_blocks * cols;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_col_reduction_2d_small<cfg, ssg_size>(
                cgh, rows, cols, row_blocks, fn, op, [](auto v) { return v; },
                identity, reinterpret_cast<ValueType*>(tmp.get_data()),
                args...);
        });
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_finalize_2d(
                cgh, cols, row_blocks, op, finalize, identity,
                reinterpret_cast<const ValueType*>(tmp.get_const_data()), 1,
                result);
        });
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generic_col_reduction_small,
                                    run_generic_col_reduction_small);


template <typename cfg, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_row_reduction_stage1(std::shared_ptr<const DpcppExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, size_type result_stride,
                                     dim<2> size, array<char>& tmp,
                                     MappedKernelArgs... args)
{
    constexpr auto wg_size = cfg::block_size;
    constexpr auto sg_size = cfg::subgroup_size;
    using subsubgroup_sizes =
        syn::value_list<int, 1, 2, 4, 8, std::min<int>(16, sg_size),
                        std::min<int>(32, sg_size), sg_size>;
    constexpr int oversubscription = 16;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto resources =
        exec->get_num_computing_units() * sg_size * oversubscription;
    auto queue = exec->get_queue();
    if (rows * cols > resources && rows < cols) {
        const auto col_blocks = ceildiv(rows * cols, resources);
        const auto required_storage = sizeof(ValueType) * col_blocks * rows;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        generic_kernel_row_reduction_2d<cfg, sg_size>(
            syn::value_list<int, sg_size>{}, exec, rows, cols, col_blocks, fn,
            op, [](auto v) { return v; }, identity,
            reinterpret_cast<ValueType*>(tmp.get_data()), 1, args...);
        queue->submit([&](sycl::handler& cgh) {
            generic_kernel_reduction_finalize_2d(
                cgh, rows, col_blocks, op, finalize, identity,
                reinterpret_cast<const ValueType*>(tmp.get_const_data()),
                static_cast<int64>(result_stride), result);
        });
    } else {
        select_generic_kernel_row_reduction_2d(
            subsubgroup_sizes(),
            [cols](int compiled_ssg_size) {
                return compiled_ssg_size >= cols ||
                       compiled_ssg_size == sg_size;
            },
            syn::value_list<int>(), syn::type_list<cfg>(), exec, rows, cols, 1,
            fn, op, finalize, identity, result,
            static_cast<int64>(result_stride), args...);
    }
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(
    select_kernel_row_reduction_stage1, run_kernel_row_reduction_stage1,
    DCFG_1D);


template <typename cfg, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_col_reduction_stage1(std::shared_ptr<const DpcppExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, dim<2> size,
                                     array<char>& tmp, MappedKernelArgs... args)
{
    constexpr auto wg_size = cfg::block_size;
    constexpr auto sg_size = cfg::subgroup_size;
    using subsubgroup_sizes =
        syn::value_list<int, 1, 2, 4, 8, std::min<int>(16, sg_size),
                        std::min<int>(32, sg_size), sg_size>;
    constexpr int oversubscription = 16;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto max_blocks =
        exec->get_num_computing_units() * sg_size * oversubscription / wg_size;
    if (cols <= sg_size) {
        select_generic_col_reduction_small(
            subsubgroup_sizes(),
            [cols](int compiled_ssg_size) {
                return compiled_ssg_size >= cols ||
                       compiled_ssg_size == sg_size;
            },
            syn::value_list<int>(), syn::type_list<cfg>(), exec, max_blocks, fn,
            op, finalize, identity, result, size, tmp, args...);
    } else {
        const auto col_blocks = ceildiv(cols, sg_size);
        const auto row_blocks = ceildiv(
            std::min<int64>(ceildiv(rows * sg_size, wg_size), max_blocks),
            col_blocks);
        auto queue = exec->get_queue();
        if (row_blocks <= 1) {
            queue->submit([&](sycl::handler& cgh) {
                generic_kernel_col_reduction_2d_blocked<cfg>(
                    cgh, rows, cols, 1, col_blocks, fn, op, finalize, identity,
                    result, args...);
            });
        } else {
            const auto required_storage = sizeof(ValueType) * row_blocks * cols;
            if (tmp.get_size() < required_storage) {
                tmp.resize_and_reset(required_storage);
            }
            queue->submit([&](sycl::handler& cgh) {
                generic_kernel_col_reduction_2d_blocked<cfg>(
                    cgh, rows, cols, row_blocks, col_blocks, fn, op,
                    [](auto v) { return v; }, identity,
                    reinterpret_cast<ValueType*>(tmp.get_data()), args...);
            });
            queue->submit([&](sycl::handler& cgh) {
                generic_kernel_reduction_finalize_2d(
                    cgh, cols, row_blocks, op, finalize, identity,
                    reinterpret_cast<const ValueType*>(tmp.get_const_data()), 1,
                    result);
            });
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_CONFIG_SELECTION_TOTYPE(
    select_kernel_col_reduction_stage1, run_kernel_col_reduction_stage1,
    DCFG_1D);


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_row_reduction_cached(std::shared_ptr<const DpcppExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, size_type result_stride,
                                     dim<2> size, array<char>& tmp,
                                     KernelArgs&&... args)
{
    const auto desired_cfg = get_first_cfg(
        as_array(dcfg_1d_list_simple_reduction), [&](std::uint32_t cfg) {
            return validate(exec->get_queue(), DCFG_1D::decode<0>(cfg),
                            DCFG_1D::decode<1>(cfg));
        });
    select_kernel_row_reduction_stage1(
        dcfg_1d_list_simple_reduction,
        [&](std::uint32_t cfg) { return cfg == desired_cfg; },
        syn::value_list<bool>(), syn::value_list<int>(),
        syn::value_list<size_type>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, result_stride, size, tmp,
        map_to_device(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction_cached(std::shared_ptr<const DpcppExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, dim<2> size,
                                     array<char>& tmp, KernelArgs&&... args)
{
    const auto desired_cfg = get_first_cfg(
        as_array(dcfg_1d_list_simple_reduction), [&](std::uint32_t cfg) {
            return validate(exec->get_queue(), DCFG_1D::decode<0>(cfg),
                            DCFG_1D::decode<1>(cfg));
        });
    select_kernel_col_reduction_stage1(
        dcfg_1d_list_simple_reduction,
        [&](std::uint32_t cfg) { return cfg == desired_cfg; },
        syn::value_list<bool>(), syn::value_list<int>(),
        syn::value_list<size_type>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, size, tmp, map_to_device(args)...);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
