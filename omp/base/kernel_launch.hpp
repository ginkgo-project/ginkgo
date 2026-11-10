// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#error \
    "This file can only be used from inside common/unified/base/kernel_launch.hpp"
#endif


#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "core/synthesizer/implementation_selection.hpp"

#ifdef GKO_OMP_HIGHWAY
#include <omp.h>

#include "omp/base/kernel_launch_simd.hpp"
#endif


namespace gko {
namespace kernels {
namespace omp {


namespace device_std = std;


#ifdef GKO_OMP_HIGHWAY


/**
 * @internal
 * Support machinery for run_kernel_simd. Kept out of gko::kernels::omp itself:
 * bodies compile in nested namespaces, so names at that scope are visible to
 * ordinary lookup from every kernel -- not something `hn` should be.
 */
namespace simd_detail {


namespace hn = hwy::HWY_NAMESPACE;


/** @internal The Highway descriptor used for a given value type. */
template <typename T>
using simd_tag = hn::ScalableTag<T>;

/** @internal Lane count of a descriptor, in the launcher's index type. */
template <typename D>
int64 simd_lanes(D d)
{
    return static_cast<int64>(hn::Lanes(d));
}


/**
 * @internal
 * The value type an argument contributes to a launch's shared value type.
 * Scalars contribute too: a differing ScalarType must disable the SIMD path,
 * since a broadcast rounds alpha before the multiply while the scalar path
 * promotes and rounds once. Non-floating-point operands map to void and stay
 * neutral.
 */
template <typename T>
struct simd_accessor_value_type {
    using type = std::conditional_t<std::is_floating_point_v<T>, T, void>;
};

template <typename T>
struct simd_accessor_value_type<matrix_accessor<T>> {
    using type = std::remove_const_t<T>;
};

template <typename T>
struct simd_accessor_value_type<T*> {
    using type =
        std::conditional_t<std::is_floating_point_v<std::remove_const_t<T>>,
                           std::remove_const_t<T>, void>;
};


template <typename A, typename B>
struct simd_combine_value_type {
    using type = void;
};
template <typename A>
struct simd_combine_value_type<A, A> {
    using type = A;
};
template <typename A>
struct simd_combine_value_type<A, void> {
    using type = A;
};
template <typename B>
struct simd_combine_value_type<void, B> {
    using type = B;
};
template <>
struct simd_combine_value_type<void, void> {
    using type = void;
};

/**
 * @internal
 * Value type shared by every contributing argument, or void if they disagree.
 * Disagreement excludes mixed-precision kernels such as dense::copy.
 */
template <typename... Args>
struct simd_value_type_of {
    using type = void;
};

template <typename Arg, typename... Rest>
struct simd_value_type_of<Arg, Rest...> {
    using type = typename simd_combine_value_type<
        typename simd_accessor_value_type<std::decay_t<Arg>>::type,
        typename simd_value_type_of<Rest...>::type>::type;
};

template <typename... Args>
using simd_value_type_of_t = typename simd_value_type_of<Args...>::type;


template <typename T>
inline constexpr bool simd_is_accessor = false;

template <typename T>
inline constexpr bool simd_is_accessor<matrix_accessor<T>> = true;

/**
 * @internal
 * True if at least one argument is a dense accessor. Without it, a launch whose
 * only floating-point argument is a scalar would look eligible and call the
 * body rows/lanes times instead of rows times.
 */
template <typename... Args>
inline constexpr bool simd_has_accessor =
    (simd_is_accessor<std::decay_t<Args>> || ...);


/** @internal Only real float and double take the SIMD path. */
template <typename T>
inline constexpr bool simd_supported_value_type =
    std::is_same_v<T, float> || std::is_same_v<T, double>;


template <typename T>
bool simd_arg_unit_stride(const matrix_accessor<T>& acc)
{
    return acc.stride == 1;
}

template <typename X>
bool simd_arg_unit_stride(const X&)
{
    return true;
}

template <typename... Args>
bool all_unit_stride(const Args&... args)
{
    return (simd_arg_unit_stride(args) && ...);
}


template <typename D, typename Access>
struct simd_accessor {
    hn::TFromD<D>* values;

    simd::ref<D, Access> operator()(simd::index i, int64) const
    {
        return {values + i.base, i.count};
    }
};

template <typename D, typename Access>
struct simd_const_accessor {
    const hn::TFromD<D>* values;

    simd::ref_leaf<D, Access> operator()(simd::index i, int64) const
    {
        return {values + i.base, i.count};
    }
};


/**
 * @internal
 * Replaces dense accessors with SIMD proxies. Everything else passes through
 * unchanged, which is why `stop[col].has_stopped()` stays a bool at cols == 1.
 */
template <typename D, typename Access, typename T>
auto simd_wrap(matrix_accessor<T> acc)
{
    if constexpr (std::is_const_v<T>) {
        return simd_const_accessor<D, Access>{acc.values};
    } else {
        return simd_accessor<D, Access>{acc.values};
    }
}

template <typename D, typename Access, typename X>
X simd_wrap(X x)
{
    return x;
}


/**
 * @internal
 * Splits [0, rows) across the current OMP team with every boundary a multiple
 * of `align` elements from the accessor's base pointer. A unit-stride submatrix
 * view can offset `values`, so this bounds false sharing without eliminating
 * it. Blocks are handed out as evenly as the count allows.
 *
 * This overload takes the participant count and caller index explicitly. The
 * partition covers [0, rows) exactly once across `num_threads`, so whatever
 * condition decides who calls it must use that same `num_threads`.
 */
inline std::pair<int64, int64> aligned_thread_range(int64 rows, int64 align,
                                                    int64 num_threads,
                                                    int64 thread_id)
{
    const auto blocks = ceildiv(rows, align);
    const auto blocks_per_thread = blocks / num_threads;
    const auto threads_with_extra_block = blocks % num_threads;
    const auto begin_block = thread_id * blocks_per_thread +
                             std::min(thread_id, threads_with_extra_block);
    const auto end_block = begin_block + blocks_per_thread +
                           (thread_id < threads_with_extra_block ? 1 : 0);
    return {std::min(rows, begin_block * align),
            std::min(rows, end_block * align)};
}


/**
 * @internal
 * Partitions across the whole current team.
 * @see aligned_thread_range(int64, int64, int64, int64)
 */
inline std::pair<int64, int64> aligned_thread_range(int64 rows, int64 align)
{
    return aligned_thread_range(rows, align,
                                std::max<int64>(omp_get_num_threads(), 1),
                                static_cast<int64>(omp_get_thread_num()));
}


/** @internal Cache line size assumed for the thread block granularity. */
constexpr int64 cache_line_bytes = 64;


/** @internal Vector lanes, rounded up to a full cache line. */
template <typename T>
int64 simd_chunk_alignment(int64 lanes)
{
    static_assert(sizeof(T) <= cache_line_bytes,
                  "a value type must fit into a cache line");
    constexpr int64 cache_line_elems =
        cache_line_bytes / static_cast<int64>(sizeof(T));
    return std::max<int64>(lanes, cache_line_elems);
}


}  // namespace simd_detail


#endif  // GKO_OMP_HIGHWAY


namespace {


template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_impl(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     size_type size, MappedKernelArgs... args)
{
#pragma omp parallel for
    for (int64 i = 0; i < static_cast<int64>(size); i++) {
        [&]() { fn(i, args...); }();
    }
}


template <int block_size, int remainder_cols, typename KernelFunction,
          typename... MappedKernelArgs>
void run_kernel_sized_impl(syn::value_list<int, remainder_cols>,
                           std::shared_ptr<const OmpExecutor> exec,
                           KernelFunction fn, dim<2> size,
                           MappedKernelArgs... args)
{
    const auto rows = static_cast<int64>(size[0]);
    const auto cols = static_cast<int64>(size[1]);
    static_assert(remainder_cols < block_size, "remainder too large");
    const auto rounded_cols = cols / block_size * block_size;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
    if (rounded_cols == 0 || cols == block_size) {
        // we group all sizes <= block_size here and unroll explicitly
        constexpr auto local_cols =
            remainder_cols == 0 ? block_size : remainder_cols;
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
#pragma unroll
            for (int64 col = 0; col < local_cols; col++) {
                [&]() { fn(row, col, args...); }();
            }
        }
    } else {
        // we operate in block_size blocks plus an explicitly unrolled remainder
#pragma omp parallel for
        for (int64 row = 0; row < rows; row++) {
            for (int64 base_col = 0; base_col < rounded_cols;
                 base_col += block_size) {
#pragma unroll
                for (int64 i = 0; i < block_size; i++) {
                    [&]() { fn(row, base_col + i, args...); }();
                }
            }
#pragma unroll
            for (int64 i = 0; i < remainder_cols; i++) {
                [&]() { fn(row, rounded_cols + i, args...); }();
            }
        }
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_run_kernel_sized,
                                    run_kernel_sized_impl);


template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_impl(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     dim<2> size, MappedKernelArgs... args)
{
    const auto cols = static_cast<int64>(size[1]);
    constexpr int block_size = 8;
    using remainders = syn::as_list<syn::range<0, block_size, 1>>;

    if (cols <= 0) {
        return;
    }
    select_run_kernel_sized(
        remainders(),
        [&](int remainder) { return remainder == cols % block_size; },
        syn::value_list<int, block_size>(), syn::type_list<>(), exec, fn, size,
        args...);
}


#ifdef GKO_OMP_HIGHWAY

template <typename KernelFunction, typename... MappedKernelArgs>
bool try_run_kernel_simd_impl(std::shared_ptr<const OmpExecutor> exec,
                              KernelFunction fn, dim<2> size,
                              MappedKernelArgs... args)
{
    using value_type = simd_detail::simd_value_type_of_t<MappedKernelArgs...>;
    if constexpr (simd_detail::simd_has_accessor<MappedKernelArgs...> &&
                  !std::is_void_v<value_type> &&
                  simd_detail::simd_supported_value_type<value_type>) {
        if (size[1] != 1 || !simd_detail::all_unit_stride(args...)) {
            return false;
        }
        const auto rows = static_cast<int64>(size[0]);
        using D = simd_detail::simd_tag<value_type>;
        const D d;
        const auto lanes = simd_detail::simd_lanes(d);
        const auto align = simd_detail::simd_chunk_alignment<value_type>(lanes);
#pragma omp parallel
        {
            const auto range = simd_detail::aligned_thread_range(rows, align);
            auto i = range.first;
            for (; i + lanes <= range.second; i += lanes) {
                fn(simd::index{i, static_cast<std::size_t>(lanes)}, int64{0},
                   simd_detail::simd_wrap<D, simd::full_access>(args)...);
            }
            if (i < range.second) {
                fn(simd::index{i, static_cast<std::size_t>(range.second - i)},
                   int64{0},
                   simd_detail::simd_wrap<D, simd::partial_access>(args)...);
            }
        }
        return true;
    } else {
        return false;
    }
}

#endif  // GKO_OMP_HIGHWAY


/**
 * @internal
 * Shared tail of run_kernel_simd and run_kernel_solver_simd. Takes already
 * mapped arguments so the mapping runs once per launch, not once per path.
 */
template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_simd_dispatch(std::shared_ptr<const OmpExecutor> exec,
                              KernelFunction fn, dim<2> size,
                              MappedKernelArgs... args)
{
#ifdef GKO_OMP_HIGHWAY
    if (try_run_kernel_simd_impl(exec, fn, size, args...)) {
        return;
    }
#endif
    run_kernel_impl(exec, fn, size, args...);
}


}  // namespace


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs&&... args)
{
    run_kernel_impl(exec, fn, size, map_to_device(args)...);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs&&... args)
{
    run_kernel_impl(exec, fn, size, map_to_device(args)...);
}


/**
 * @internal
 * Elementwise launcher that vectorizes when the runtime shape allows it.
 * Only opt in for bodies safe to vectorize across rows; one that inspects the
 * row index will not compile against simd::index, which is deliberate.
 * WARNING: an accessor result here is a lazy alias, not a copy as on the
 * scalar path -- see `simd::ref`.
 */
template <typename KernelFunction, typename... KernelArgs>
void run_kernel_simd(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     dim<2> size, KernelArgs&&... args)
{
    run_kernel_simd_dispatch(exec, fn, size, map_to_device(args)...);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko
