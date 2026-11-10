// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#define GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_


#include <type_traits>

#include "common/unified/base/kernel_launch.hpp"


#define GKO_KERNEL_REDUCE_SUM(ValueType)                            \
    gko::kernels::GKO_DEVICE_NAMESPACE::reduce_sum_op<ValueType>{}, \
        [] GKO_KERNEL(auto a) { return a; }, ValueType              \
    {}
#define GKO_KERNEL_REDUCE_MAX(ValueType)                            \
    gko::kernels::GKO_DEVICE_NAMESPACE::reduce_max_op<ValueType>{}, \
        [] GKO_KERNEL(auto a) { return a; }, ValueType              \
    {}


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * Highway-free half of the gate: a functor advertises its vector form with a
 * marker typedef, declared only alongside *both* `vector_op` and
 * `reduce_lanes`. Probing the marker rather than the members keeps this trait
 * free of Highway types, so it stays well-formed in a TU without Highway.
 */
template <typename Op, typename = void>
struct has_simd_reduction_marker_impl : std::false_type {};

template <typename Op>
struct has_simd_reduction_marker_impl<
    Op, std::void_t<typename Op::gko_simd_reduction>> : std::true_type {};


/**
 * @internal
 * Highway-dependent half: checks both member templates the launcher calls are
 * present, so an op carrying only one falls back to the scalar path instead of
 * failing inside the launcher. Guarded on `GKO_COMPILING_OMP` too, so it never
 * names a Highway type in a TU without Highway headers.
 */
#if defined(GKO_OMP_HIGHWAY) && defined(GKO_COMPILING_OMP)
template <typename Op, typename = void>
struct has_simd_reduction_members_impl : std::false_type {};

template <typename Op>
struct has_simd_reduction_members_impl<
    Op, std::void_t<decltype(&Op::template vector_op<
                             hwy::HWY_NAMESPACE::ScalableTag<double>>),
                    decltype(&Op::template reduce_lanes<
                             hwy::HWY_NAMESPACE::ScalableTag<double>>)>>
    : std::true_type {};
#else
template <typename Op, typename = void>
struct has_simd_reduction_members_impl : std::false_type {};
#endif


/**
 * @internal
 * True when a reduction op carries a vector form; a raw lambda takes the scalar
 * path. A `ScalableTag<double>` probe also answers correctly for a `float` op,
 * because the functors' `ValueType` is unused.
 */
template <typename Op>
inline constexpr bool has_simd_reduction =
    std::conjunction_v<has_simd_reduction_marker_impl<Op>,
                       has_simd_reduction_members_impl<Op>>;


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


// has_simd_reduction must be declared before this include: the OMP launcher
// names it unqualified inside a template, so it has to be visible at the
// template's point of definition.
#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
#include "common/cuda_hip/base/kernel_launch_reduction.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/base/kernel_launch_reduction.dp.hpp"
#elif defined(GKO_COMPILING_OMP)
#include "omp/base/kernel_launch_reduction.hpp"
#endif


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


/**
 * @internal
 * Reduction operations are functors rather than lambdas so the OMP SIMD path
 * can pick up a vector form alongside the scalar one. `ValueType` is unused --
 * every member deduces from its own parameters -- and kept only because the
 * GKO_KERNEL_REDUCE_* call sites spell it.
 */
template <typename ValueType>
struct reduce_sum_op {
    // Two independent template parameters, because the lambda this replaces
    // deduced each operand separately -- ell_kernels reduces an `int` into an
    // `IndexType`. The return type must be deduced `auto`, not
    // `decltype(a + b)`: for `a > b ? a : b` that yields `U&`, a reference to a
    // by-value parameter.
    template <typename U, typename V>
    GKO_INLINE GKO_ATTRIBUTES auto operator()(U a, V b) const
    {
        return a + b;
    }

#ifdef GKO_OMP_HIGHWAY
    // Highway-free marker read by has_simd_reduction. Declare it only next to
    // BOTH members below: it is the advertisement that this op implements the
    // whole vector reduction interface.
    using gko_simd_reduction = void;

    template <typename D>
    static hwy::HWY_NAMESPACE::Vec<D> vector_op(D, hwy::HWY_NAMESPACE::Vec<D> a,
                                                hwy::HWY_NAMESPACE::Vec<D> b)
    {
        return hwy::HWY_NAMESPACE::Add(a, b);
    }

    template <typename D>
    static hwy::HWY_NAMESPACE::TFromD<D> reduce_lanes(
        D d, hwy::HWY_NAMESPACE::Vec<D> v)
    {
        return hwy::HWY_NAMESPACE::ReduceSum(d, v);
    }
#endif
};

// `ValueType` is unused here too; see reduce_sum_op above.
template <typename ValueType>
struct reduce_max_op {
    // See reduce_sum_op above for why this takes two independent template
    // parameters and returns a deduced `auto` rather than a trailing decltype.
    template <typename U, typename V>
    GKO_INLINE GKO_ATTRIBUTES auto operator()(U a, V b) const
    {
        return a > b ? a : b;
    }

#ifdef GKO_OMP_HIGHWAY
    // @see reduce_sum_op::gko_simd_reduction
    using gko_simd_reduction = void;

    template <typename D>
    static hwy::HWY_NAMESPACE::Vec<D> vector_op(D, hwy::HWY_NAMESPACE::Vec<D> a,
                                                hwy::HWY_NAMESPACE::Vec<D> b)
    {
        return hwy::HWY_NAMESPACE::Max(a, b);
    }

    template <typename D>
    static hwy::HWY_NAMESPACE::TFromD<D> reduce_lanes(
        D d, hwy::HWY_NAMESPACE::Vec<D> v)
    {
        return hwy::HWY_NAMESPACE::ReduceMax(d, v);
    }
#endif
};


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const DefaultExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType identity,
                          ValueType* result, size_type size,
                          KernelArgs&&... args)
{
    array<char> cache{exec};
    run_kernel_reduction_cached(exec, fn, op, finalize, identity, result, size,
                                cache, std::forward<KernelArgs>(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_reduction(std::shared_ptr<const DefaultExecutor> exec,
                          KernelFunction fn, ReductionOp op,
                          FinalizeOp finalize, ValueType identity,
                          ValueType* result, dim<2> size, KernelArgs&&... args)
{
    array<char> cache{exec};
    run_kernel_reduction_cached(exec, fn, op, finalize, identity, result, size,
                                cache, std::forward<KernelArgs>(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_row_reduction(std::shared_ptr<const DefaultExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType identity,
                              ValueType* result, size_type result_stride,
                              dim<2> size, KernelArgs&&... args)
{
    array<char> cache{exec};
    run_kernel_row_reduction_cached(exec, fn, op, finalize, identity, result,
                                    result_stride, size, cache,
                                    std::forward<KernelArgs>(args)...);
}


template <typename ValueType, typename KernelFunction, typename ReductionOp,
          typename FinalizeOp, typename... KernelArgs>
void run_kernel_col_reduction(std::shared_ptr<const DefaultExecutor> exec,
                              KernelFunction fn, ReductionOp op,
                              FinalizeOp finalize, ValueType identity,
                              ValueType* result, dim<2> size,
                              KernelArgs&&... args)
{
    array<char> cache{exec};
    run_kernel_col_reduction_cached(exec, fn, op, finalize, identity, result,
                                    size, cache,
                                    std::forward<KernelArgs>(args)...);
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
