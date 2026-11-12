// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#define GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_


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
 * Reduction operations are functors rather than lambdas so the OMP SIMD path
 * can recognise them: it matches on the type to pick up a vector form. The
 * vector form itself lives in omp/base/kernel_launch_reduction.hpp, which
 * keeps every Highway type out of this header and leaves these classes with
 * one definition in every translation unit. `ValueType` is unused -- each
 * member deduces from its own parameters -- and kept only because the
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
};


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


// The reduction ops must be complete before this include: the OMP backend
// specialises simd_reduction on them to supply their vector form.
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
