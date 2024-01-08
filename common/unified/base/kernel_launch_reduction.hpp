// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_
#define GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_REDUCTION_HPP_


#include "common/unified/base/kernel_launch.hpp"


#define GKO_KERNEL_REDUCE_SUM(ValueType)               \
    [] GKO_KERNEL(auto a, auto b) { return a + b; },   \
        [] GKO_KERNEL(auto a) { return a; }, ValueType \
    {}
#define GKO_KERNEL_REDUCE_MAX(ValueType)                     \
    [] GKO_KERNEL(auto a, auto b) { return a > b ? a : b; }, \
        [] GKO_KERNEL(auto a) { return a; }, ValueType       \
    {}


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/kernel_launch_reduction.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/kernel_launch_reduction.hip.hpp"
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
