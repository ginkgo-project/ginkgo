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

#ifndef GKO_HIP_BASE_KERNEL_LAUNCH_HIP_HPP_
#define GKO_HIP_BASE_KERNEL_LAUNCH_HIP_HPP_


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


#ifdef GKO_KERNEL
#error "Only one kernel_launch.hpp file can be included at a time."
#else
#define GKO_KERNEL __device__
#endif


namespace gko {
namespace kernels {
namespace hip {


constexpr auto default_block_size = 512;


template <typename ValueType>
struct matrix_accessor {
    ValueType *data;
    size_type stride;

    __device__ ValueType &operator()(size_type row, size_type col)
    {
        return data[row * stride + col];
    }

    __device__ ValueType &operator[](size_type idx) { return data[idx]; }
};


template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_1d(
    size_type size, KernelFunction fn, KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat();
    if (tidx >= size) {
        return;
    }
    fn(tidx, args...);
}


template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_2d(
    size_type rows, size_type cols, KernelFunction fn, KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat();
    auto col = tidx % cols;
    auto row = tidx / cols;
    if (row >= rows) {
        return;
    }
    fn(row, col, args...);
}


template <typename ValueType>
matrix_accessor<hip_type<ValueType>> map_to_device(
    matrix::Dense<ValueType> *mtx)
{
    return {as_hip_type(mtx->get_values()), mtx->get_stride()};
}


template <typename ValueType>
matrix_accessor<const hip_type<ValueType>> map_to_device(
    const matrix::Dense<ValueType> *mtx)
{
    return {as_hip_type(mtx->get_const_values()), mtx->get_stride()};
}


template <typename ValueType>
typename std::enable_if<std::is_arithmetic<ValueType>::value,
                        hip_type<ValueType>>::type *
map_to_device(ValueType *data)
{
    return as_hip_type(data);
}


template <typename ValueType>
hip_type<std::complex<ValueType>> *map_to_device(std::complex<ValueType> *data)
{
    return as_hip_type(data);
}


template <typename ValueType>
const hip_type<std::complex<ValueType>> *map_to_device(
    const std::complex<ValueType> *data)
{
    return as_hip_type(data);
}


template <typename ValueType>
hip_type<ValueType> *map_to_device(Array<ValueType> &mtx)
{
    return as_hip_type(mtx.get_data());
}


template <typename ValueType>
const hip_type<ValueType> *map_to_device(const Array<ValueType> &mtx)
{
    return as_hip_type(mtx.get_const_data());
}


}  // namespace hip
}  // namespace kernels


template <typename KernelFunction, typename... KernelArgs>
void HipExecutor::run_kernel(KernelFunction fn, size_type size,
                             KernelArgs &&... args) const
{
    constexpr auto block_size = kernels::hip::default_block_size;
    auto num_blocks = ceildiv(size, block_size);
    hipLaunchKernelGGL(kernels::hip::generic_kernel_1d, num_blocks, block_size,
                       0, 0, size, fn, kernels::hip::map_to_device(args)...);
}


template <typename KernelFunction, typename... KernelArgs>
void HipExecutor::run_kernel(KernelFunction fn, dim<2> size,
                             KernelArgs &&... args) const
{
    constexpr auto block_size = kernels::hip::default_block_size;
    auto num_blocks = ceildiv(size[0] * size[1], block_size);
    hipLaunchKernelGGL(kernels::hip::generic_kernel_2d, num_blocks, block_size,
                       0, 0, size[0], size[1], fn,
                       kernels::hip::map_to_device(args)...);
}


}  // namespace gko

#endif  // GKO_HIP_BASE_KERNEL_LAUNCH_HIP_HPP_
