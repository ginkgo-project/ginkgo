// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch_reduction.hpp"
#endif


#include <algorithm>
#include <numeric>
#include <tuple>

#include <omp.h>


namespace gko {
namespace kernels {
namespace omp {


// how many more reduction tasks we launch relative to the number of threads
constexpr int reduction_kernel_oversubscription = 4;

// `num_threads(...)` is a request, not a guarantee. Seed the partials and
// derive each range from the team actually granted.

// Inside parallel regions, the kernel arguments are only expanded within a
// `[&]` lambda: icpx 2024 reads floating-point values that are not the first
// entry of a parameter pack expanded directly in a parallel region as zero.


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
    const auto required_storage = sizeof(ValueType) * num_threads;
    if (tmp.get_size() < required_storage) {
        tmp.resize_and_reset(required_storage);
    }
    const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
    std::fill_n(partial, num_threads, identity);
#pragma omp parallel num_threads(num_threads)
    {
        const auto thread_id = omp_get_thread_num();
        const auto team_size = std::min<int64>(
            std::max<int64>(omp_get_num_threads(), 1), num_threads);
        const auto work_per_thread =
            ceildiv(ssize, std::max<int64>(team_size, 1));
        if (thread_id < team_size) {
            const auto begin = thread_id * work_per_thread;
            const auto end = std::min(ssize, begin + work_per_thread);

            auto local_partial = identity;
            for (auto i = begin; i < end; i++) {
                local_partial = op(local_partial, [&]() {
                    return fn(i, map_to_device(args)...);
                }());
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
    const auto required_storage = sizeof(ValueType) * num_threads;
    if (tmp.get_size() < required_storage) {
        tmp.resize_and_reset(required_storage);
    }
    const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
    std::fill_n(partial, num_threads, identity);
    static_assert(remainder_cols < block_size, "remainder too large");
    const auto rounded_cols = cols / block_size * block_size;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
#pragma omp parallel num_threads(num_threads)
    {
        const auto thread_id = omp_get_thread_num();
        const auto team_size = std::min<int64>(
            std::max<int64>(omp_get_num_threads(), 1), num_threads);
        const auto work_per_thread =
            ceildiv(rows, std::max<int64>(team_size, 1));
        if (thread_id < team_size) {
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
                        local_partial = op(local_partial, [&]() {
                            return fn(row, col, args...);
                        }());
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
                            local_partial = op(local_partial, [&]() {
                                return fn(row, base_col + i, args...);
                            }());
                        }
                    }
#pragma unroll
                    for (int64 i = 0; i < remainder_cols; i++) {
                        local_partial = op(local_partial, [&]() {
                            return fn(row, rounded_cols + i, args...);
                        }());
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
        const auto temp_elems_per_row = num_threads;
        const auto required_storage =
            sizeof(ValueType) * rows * temp_elems_per_row;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
        std::fill_n(partial, rows * temp_elems_per_row, identity);
#pragma omp parallel num_threads(num_threads)
        {
            const auto thread_id = static_cast<int64>(omp_get_thread_num());
            const auto team_size = std::min<int64>(
                std::max<int64>(omp_get_num_threads(), 1), num_threads);
            const auto work_per_thread =
                ceildiv(cols, std::max<int64>(team_size, 1));
            if (thread_id < team_size) {
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
            [&] {
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
            }();
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
            [&] {
                const auto col_block = i % num_col_blocks;
                const auto row_block = i / num_col_blocks;
                const auto begin = row_block * rows_per_thread;
                const auto end = std::min(begin + rows_per_thread, rows);
                const auto base_col = col_block * block_size;
                const auto identity_fn = [](auto i) { return i; };
                if (base_col + block_size <= cols) {
                    run_kernel_col_reduction_sized_block_impl<block_size>(
                        fn, op, identity_fn, identity,
                        partial + cols * row_block, begin, end, base_col,
                        args...);
                } else {
                    run_kernel_col_reduction_sized_block_impl<remainder_cols>(
                        fn, op, identity_fn, identity,
                        partial + cols * row_block, begin, end, base_col,
                        args...);
                }
            }();
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


#ifdef GKO_OMP_HIGHWAY

/**
 * @internal
 * Reduces `rows` contiguous elements at cols == 1.
 *
 * PRECONDITION: `identity` must survive lane replication, i.e.
 * `reduce_lanes(d, Set(d, identity)) == identity`. The accumulators are seeded
 * with `identity` in every lane, so one that does not survive the fold is
 * counted per lane rather than per thread. True for max/min always, for sum
 * only when `identity == 0`; asserted below, as `identity` is a runtime value.
 *
 * Four named accumulators rather than an array, and Set-initialized: Vec is
 * sizeless on SVE/RVV and not default-constructible.
 */
template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... MappedKernelArgs>
bool try_run_kernel_reduction_simd_impl(std::shared_ptr<const OmpExecutor> exec,
                                        KernelFunction fn, ReductionOp op,
                                        FinalizeOp finalize, ValueType identity,
                                        ValueType* result, dim<2> size,
                                        array<char>& tmp,
                                        MappedKernelArgs... args)
{
    using value_type = simd_detail::simd_value_type_of_t<MappedKernelArgs...>;
    if constexpr (simd_detail::simd_has_accessor<MappedKernelArgs...> &&
                  !std::is_void_v<value_type> &&
                  simd_detail::simd_supported_value_type<value_type> &&
                  std::is_same_v<value_type, ValueType> &&
                  has_simd_reduction<ReductionOp>) {
        if (size[1] != 1 || !simd_detail::all_unit_stride(args...)) {
            return false;
        }
        const auto rows = static_cast<int64>(size[0]);
        using D = simd_detail::simd_tag<value_type>;
        const D d;
        const auto lanes = simd_detail::simd_lanes(d);
        const auto num_threads =
            std::min<int64>(omp_get_max_threads(), std::max<int64>(rows, 1));
        const auto required_storage = sizeof(ValueType) * num_threads;
        if (tmp.get_size() < required_storage) {
            tmp.resize_and_reset(required_storage);
        }
        const auto partial = reinterpret_cast<ValueType*>(tmp.get_data());
        const auto align = simd_detail::simd_chunk_alignment<value_type>(lanes);
        // See the precondition in this function's doc comment. Debug-only:
        // GKO_ASSERT is plain assert(), so this disappears entirely under
        // NDEBUG. It is an equality on the identity itself, not on a computed
        // result, so it holds exactly for the legitimate cases -- ReduceMax of
        // a vector of -inf is -inf, ReduceSum of a vector of 0 is 0.
        GKO_ASSERT(ReductionOp::reduce_lanes(
                       d, simd_detail::hn::Set(d, identity)) == identity);
        // `num_threads` is a request; the runtime may grant a smaller team.
        // Seeding keeps partial[team_size, num_threads) neutral, since the
        // final accumulate reads all `num_threads` entries.
        std::fill_n(partial, num_threads, identity);
#pragma omp parallel num_threads(num_threads)
        {
            const auto thread_id = static_cast<int64>(omp_get_thread_num());
            // Guard and partition must agree on who owns work, or a thread
            // given a range but skipped by the guard drops its rows: both use
            // `team_size`.
            const auto team_size = std::min<int64>(
                std::max<int64>(omp_get_num_threads(), 1), num_threads);
            if (thread_id < team_size) {
                const auto range = simd_detail::aligned_thread_range(
                    rows, align, team_size, thread_id);
                auto acc0 = simd_detail::hn::Set(d, identity);
                auto acc1 = simd_detail::hn::Set(d, identity);
                auto acc2 = simd_detail::hn::Set(d, identity);
                auto acc3 = simd_detail::hn::Set(d, identity);
                auto i = range.first;
                const auto end = range.second;
                const auto lane_count = static_cast<std::size_t>(lanes);
                // Index-independent, so built once: read through `args` the
                // shared OMP capture is reloaded on every iteration, which
                // costs one base pointer plus four address computations.
                const auto wrapped = std::make_tuple(
                    simd_detail::simd_wrap<D, simd::full_access>(args)...);
                const auto at = [&](auto idx) {
                    return std::apply(
                        [&](auto... w) {
                            return simd::eval_to_vec(d,
                                                     fn(idx, int64{0}, w...));
                        },
                        wrapped);
                };
                for (; i + 4 * lanes <= end; i += 4 * lanes) {
                    acc0 = ReductionOp::vector_op(
                        d, acc0, at(simd::index{i, lane_count}));
                    acc1 = ReductionOp::vector_op(
                        d, acc1, at(simd::index{i + lanes, lane_count}));
                    acc2 = ReductionOp::vector_op(
                        d, acc2, at(simd::index{i + 2 * lanes, lane_count}));
                    acc3 = ReductionOp::vector_op(
                        d, acc3, at(simd::index{i + 3 * lanes, lane_count}));
                }
                for (; i + lanes <= end; i += lanes) {
                    acc0 = ReductionOp::vector_op(
                        d, acc0, at(simd::index{i, lane_count}));
                }
                if (i < end) {
                    const auto rem = static_cast<std::size_t>(end - i);
                    const auto v = simd::eval_to_vec(
                        d, fn(simd::index{i, rem}, int64{0},
                              simd_detail::simd_wrap<D, simd::partial_access>(
                                  args)...));
                    // Masked to identity rather than relying on LoadN's
                    // zero-fill: zero is the identity for sum, not for max.
                    acc1 = ReductionOp::vector_op(
                        d, acc1,
                        simd_detail::hn::IfThenElse(
                            simd_detail::hn::FirstN(d, rem), v,
                            simd_detail::hn::Set(d, identity)));
                }
                acc0 = ReductionOp::vector_op(d, acc0, acc1);
                acc2 = ReductionOp::vector_op(d, acc2, acc3);
                partial[thread_id] = ReductionOp::reduce_lanes(
                    d, ReductionOp::vector_op(d, acc0, acc2));
            }
        }
        *result = finalize(
            std::accumulate(partial, partial + num_threads, identity, op));
        return true;
    } else {
        return false;
    }
}

#endif  // GKO_OMP_HIGHWAY


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... MappedKernelArgs>
void run_kernel_col_reduction_impl(std::shared_ptr<const OmpExecutor> exec,
                                   KernelFunction fn, ReductionOp op,
                                   FinalizeOp finalize, ValueType identity,
                                   ValueType* result, dim<2> size,
                                   array<char>& tmp, MappedKernelArgs... args)
{
    constexpr auto block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;
    const auto cols = static_cast<int64>(size[1]);
    if (cols <= 0) {
        return;
    }
    select_run_kernel_col_reduction_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, op,
        finalize, identity, result, size, tmp, args...);
}


/**
 * @internal
 * Shared tail of run_kernel_col_reduction_cached_simd. Takes already mapped
 * arguments so the mapping runs once per launch, not once per path. Mirrors
 * run_kernel_simd_dispatch in omp/base/kernel_launch.hpp.
 */
template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... MappedKernelArgs>
void run_kernel_col_reduction_simd_dispatch(
    std::shared_ptr<const OmpExecutor> exec, KernelFunction fn, ReductionOp op,
    FinalizeOp finalize, ValueType identity, ValueType* result, dim<2> size,
    array<char>& tmp, MappedKernelArgs... args)
{
#ifdef GKO_OMP_HIGHWAY
    if (try_run_kernel_reduction_simd_impl(exec, fn, op, finalize, identity,
                                           result, size, tmp, args...)) {
        return;
    }
#endif
    run_kernel_col_reduction_impl(exec, fn, op, finalize, identity, result,
                                  size, tmp, args...);
}


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
    run_kernel_col_reduction_impl(exec, fn, op, finalize, identity, result,
                                  size, tmp, map_to_device(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction_cached_simd(
    std::shared_ptr<const OmpExecutor> exec, KernelFunction fn, ReductionOp op,
    FinalizeOp finalize, ValueType identity, ValueType* result, dim<2> size,
    array<char>& tmp, KernelArgs&&... args)
{
#ifdef GKO_OMP_HIGHWAY
    static_assert(has_simd_reduction<ReductionOp>,
                  "a *_simd reduction launcher needs an op with a vector form; "
                  "use run_kernel_col_reduction_cached for a raw lambda");
#endif
    run_kernel_col_reduction_simd_dispatch(exec, fn, op, finalize, identity,
                                           result, size, tmp,
                                           map_to_device(args)...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
