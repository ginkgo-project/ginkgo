// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_reduction.hpp"
#endif


#include <numeric>


#include <omp.h>


namespace gko {
namespace kernels {
namespace omp {


// how many more reduction tasks we launch relative to the number of threads
constexpr int reduction_kernel_oversubscription = 4;


namespace {


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... MappedKernelArgs>
void run_kernel_reduction_impl(std::shared_ptr<const OmpExecutor> exec,
                               KernelFunction fn, ReductionOp op,
                               FinalizeOp finalize, ValueType identity,
                               ValueType* result, size_type size,
                               array<char>& tmp, MappedKernelArgs... args)
{
    const auto ssize = static_cast<int64>(size);
    // Limit the number of threads to the number of columns
    const auto num_threads = std::min<int64>(omp_get_max_threads(), ssize);
    const auto work_per_thread =
        ceildiv(ssize, std::max<int64>(num_threads, 1));
    const auto required_storage = sizeof(ValueType) * num_threads;
    if (tmp.get_size() < required_storage) {
        tmp.resize_and_reset(required_storage);
    }
    const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
#pragma omp parallel num_threads(num_threads)
    {
        const auto thread_id = omp_get_thread_num();
        if (thread_id < num_threads) {
            const auto begin = thread_id * work_per_thread;
            const auto end = std::min(ssize, begin + work_per_thread);

            auto local_partial = identity;
            for (auto i = begin; i < end; i++) {
                local_partial =
                    op(local_partial, fn(i, map_to_device(args)...));
            }
            partial[thread_id] = local_partial;
        }
    }
    *result =
        finalize(std::accumulate(partial, partial + num_threads, identity, op));
}


template <int block_size, int remainder_cols, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_reduction_sized_impl(syn::value_list<int, remainder_cols>,
                                     std::shared_ptr<const OmpExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, dim<2> size,
                                     array<char>& tmp, MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    // Limit the number of threads to the number of columns
    const auto num_threads = std::min<int64>(omp_get_max_threads(), rows);
    const auto work_per_thread = ceildiv(rows, std::max<int64>(num_threads, 1));
    const auto required_storage = sizeof(ValueType) * num_threads;
    if (tmp.get_size() < required_storage) {
        tmp.resize_and_reset(required_storage);
    }
    const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
    static_assert(remainder_cols < block_size, "remainder too large");
    const auto rounded_cols = cols / block_size * block_size;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
#pragma omp parallel num_threads(num_threads)
    {
        const auto thread_id = omp_get_thread_num();
        if (thread_id < num_threads) {
            const auto begin = thread_id * work_per_thread;
            const auto end = std::min(rows, begin + work_per_thread);

            auto local_partial = identity;
            if (rounded_cols == 0 || cols == block_size) {
                // we group all sizes <= block_size here and unroll explicitly
                constexpr auto local_cols =
                    remainder_cols == 0 ? block_size : remainder_cols;
                for (auto row = begin; row < end; row++) {
#pragma unroll
                    for (int64 col = 0; col < local_cols; col++) {
                        local_partial =
                            op(local_partial, fn(row, col, args...));
                    }
                }
            } else {
                // we operate in block_size blocks plus an explicitly unrolled
                // remainder
                for (auto row = begin; row < end; row++) {
                    for (int64 base_col = 0; base_col < rounded_cols;
                         base_col += block_size) {
#pragma unroll
                        for (int64 i = 0; i < block_size; i++) {
                            local_partial = op(local_partial,
                                               fn(row, base_col + i, args...));
                        }
                    }
#pragma unroll
                    for (int64 i = 0; i < remainder_cols; i++) {
                        local_partial = op(local_partial,
                                           fn(row, rounded_cols + i, args...));
                    }
                }
            }
            partial[thread_id] = local_partial;
        }
    }
    *result =
        finalize(std::accumulate(partial, partial + num_threads, identity, op));
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_reduction_sized,
                                    run_kernel_reduction_sized_impl);


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction_cached(std::shared_ptr<const OmpExecutor> exec,
                                 KernelFunction fn, ReductionOp op,
                                 FinalizeOp finalize, ValueType identity,
                                 ValueType* result, size_type size,
                                 array<char>& tmp, KernelArgs&&... args)
{
    run_kernel_reduction_impl(exec, fn, op, finalize, identity, result, size,
                              tmp, map_to_device(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction_cached(std::shared_ptr<const OmpExecutor> exec,
                                 KernelFunction fn, ReductionOp op,
                                 FinalizeOp finalize, ValueType identity,
                                 ValueType* result, dim<2> size,
                                 array<char>& tmp, KernelArgs&&... args)
{
    const auto cols = static_cast<int64>(size[1]);
    constexpr int block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;

    if (cols <= 0) {
        *result = identity;
        return;
    }
    select_run_kernel_reduction_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, size, tmp, map_to_device(args)...);
}


namespace {


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... MappedKernelArgs>
void run_kernel_row_reduction_impl(std::shared_ptr<const OmpExecutor> exec,
                                   KernelFunction fn, ReductionOp op,
                                   FinalizeOp finalize, ValueType identity,
                                   ValueType* result, size_type result_stride,
                                   dim<2> size, array<char>& tmp,
                                   MappedKernelArgs... args)
{
    constexpr int block_size = 8;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto available_threads = static_cast<int64>(omp_get_max_threads());
    if (rows <= 0) {
        return;
    }
    // enough work to keep all threads busy or only very small reduction sizes
    if (rows >= reduction_kernel_oversubscription * available_threads ||
        cols < rows) {
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
            [&]() {
                auto partial = identity;
                for (int64 col = 0; col < cols; col++) {
                    partial = op(partial, fn(row, col, args...));
                }
                result[result_stride * row] = finalize(partial);
            }();
        }
    } else {
        // small number of rows and large reduction sizes: do partial sum first
        const auto num_threads = std::min<int64>(available_threads, cols);
        const auto work_per_thread =
            ceildiv(cols, std::max<int64>(num_threads, 1));
        const auto temp_elems_per_row = num_threads;
        const auto required_storage =
            sizeof(ValueType) * rows * temp_elems_per_row;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
#pragma omp parallel num_threads(num_threads)
        {
            const auto thread_id = static_cast<int64>(omp_get_thread_num());
            if (thread_id < num_threads) {
                const auto begin = thread_id * work_per_thread;
                const auto end = std::min(begin + work_per_thread, cols);
                for (int64 row = 0; row < rows; row++) {
                    auto local_partial = identity;
                    for (int64 col = begin; col < end; col++) {
                        local_partial = op(local_partial, [&]() {
                            return fn(row, col, args...);
                        }());
                    }
                    partial[row * temp_elems_per_row + thread_id] =
                        local_partial;
                }
            }
        }
        // then accumulate the partial sums and write to result
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
            [&] {
                auto local_partial = identity;
                for (int64 thread_id = 0; thread_id < temp_elems_per_row;
                     thread_id++) {
                    local_partial =
                        op(local_partial,
                           partial[row * temp_elems_per_row + thread_id]);
                }
                result[row * result_stride] = finalize(local_partial);
            }();
        }
    }
}


template <int local_cols, typename ValueType, typename KernelFunction,
          typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_col_reduction_sized_block_impl(
    KernelFunction fn, ReductionOp op, FinalizeOp finalize, ValueType identity,
    ValueType* result, int64 row_begin, int64 row_end, int64 base_col,
    MappedKernelArgs... args)
{
    std::array<ValueType, local_cols> partial;
    partial.fill(identity);
    for (auto row = row_begin; row < row_end; row++) {
#pragma unroll
        for (int64 rel_col = 0; rel_col < local_cols; rel_col++) {
            partial[rel_col] =
                op(partial[rel_col], fn(row, base_col + rel_col, args...));
        }
    }
#pragma unroll
    for (int64 rel_col = 0; rel_col < local_cols; rel_col++) {
        result[base_col + rel_col] = finalize(partial[rel_col]);
    }
}


template <int block_size, int remainder_cols, typename ValueType,
          typename KernelFunction, typename ReductionOp, typename FinalizeOp,
          typename... MappedKernelArgs>
void run_kernel_col_reduction_sized_impl(
    syn::value_list<int, remainder_cols>,
    std::shared_ptr<const OmpExecutor> exec, KernelFunction fn, ReductionOp op,
    FinalizeOp finalize, ValueType identity, ValueType* result, dim<2> size,
    array<char>& tmp, MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    const auto available_threads = static_cast<int64>(omp_get_max_threads());
    static_assert(remainder_cols < block_size, "remainder too large");
    GKO_ASSERT(cols % block_size == remainder_cols);
    const auto num_col_blocks = ceildiv(cols, block_size);
    // enough work to keep all threads busy or only very small reduction sizes
    if (cols >= reduction_kernel_oversubscription * available_threads ||
        rows < cols) {
#pragma omp parallel for
        for (int64 col_block = 0; col_block < num_col_blocks; col_block++) {
            const auto base_col = col_block * block_size;
            if (base_col + block_size <= cols) {
                run_kernel_col_reduction_sized_block_impl<block_size>(
                    fn, op, finalize, identity, result, 0, rows, base_col,
                    args...);
            } else {
                run_kernel_col_reduction_sized_block_impl<remainder_cols>(
                    fn, op, finalize, identity, result, 0, rows, base_col,
                    args...);
            }
        }
    } else {
        // number of blocks that need to be reduced afterwards
        // This reduction_size definition ensures we don't use more temporary
        // storage than the input vector
        const auto reduction_size = std::min(
            rows, ceildiv(reduction_kernel_oversubscription * available_threads,
                          std::max<int64>(cols, 1)));
        const auto rows_per_thread =
            ceildiv(rows, std::max<int64>(reduction_size, 1));
        const auto required_storage = sizeof(ValueType) * cols * reduction_size;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
#pragma omp parallel for
        for (int64 i = 0; i < reduction_size * num_col_blocks; i++) {
            const auto col_block = i % num_col_blocks;
            const auto row_block = i / num_col_blocks;
            const auto begin = row_block * rows_per_thread;
            const auto end = std::min(begin + rows_per_thread, rows);
            const auto base_col = col_block * block_size;
            const auto identity_fn = [](auto i) { return i; };
            if (base_col + block_size <= cols) {
                run_kernel_col_reduction_sized_block_impl<block_size>(
                    fn, op, identity_fn, identity, partial + cols * row_block,
                    begin, end, base_col, args...);
            } else {
                run_kernel_col_reduction_sized_block_impl<remainder_cols>(
                    fn, op, identity_fn, identity, partial + cols * row_block,
                    begin, end, base_col, args...);
            }
        }
#pragma omp parallel for
        for (int64 col = 0; col < cols; col++) {
            [&] {
                auto total = identity;
                for (int64 row_block = 0; row_block < reduction_size;
                     row_block++) {
                    total = op(total, partial[col + cols * row_block]);
                }
                result[col] = finalize(total);
            }();
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_col_reduction_sized,
                                    run_kernel_col_reduction_sized_impl);


}  // namespace


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_row_reduction_cached(std::shared_ptr<const OmpExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, size_type result_stride,
                                     dim<2> size, array<char>& tmp,
                                     KernelArgs&&... args)
{
    run_kernel_row_reduction_impl(exec, fn, op, finalize, identity, result,
                                  result_stride, size, tmp,
                                  map_to_device(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction_cached(std::shared_ptr<const OmpExecutor> exec,
                                     KernelFunction fn, ReductionOp op,
                                     FinalizeOp finalize, ValueType identity,
                                     ValueType* result, dim<2> size,
                                     array<char>& tmp, KernelArgs&&... args)
{
    constexpr auto block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    if (cols <= 0) {
        return;
    }
    select_run_kernel_col_reduction_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, size, tmp, map_to_device(args)...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
