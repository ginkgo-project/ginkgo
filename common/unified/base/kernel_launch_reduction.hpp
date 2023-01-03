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
