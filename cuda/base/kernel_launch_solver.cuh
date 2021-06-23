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

#ifndef GKO_CUDA_BASE_KERNEL_LAUNCH_SOLVER_CUH_
#define GKO_CUDA_BASE_KERNEL_LAUNCH_SOLVER_CUH_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "cuda/base/kernel_launch.cuh"


namespace gko {
namespace kernels {
namespace cuda {


template <typename ValueType>
struct default_stride_dense_wrapper {
    ValueType *data;
};


template <typename T>
struct device_unpack_solver_impl {
    using type = T;
    static __device__ __forceinline__ type unpack(T param, size_type)
    {
        return param;
    }
};

template <typename ValueType>
struct device_unpack_solver_impl<default_stride_dense_wrapper<ValueType>> {
    using type = matrix_accessor<ValueType>;
    static __device__ __forceinline__ type unpack(
        default_stride_dense_wrapper<ValueType> param, size_type default_stride)
    {
        return {param.data, default_stride};
    }
};


template <typename KernelFunction, typename... KernelArgs>
__global__ __launch_bounds__(default_block_size) void generic_kernel_2d_solver(
    size_type rows, size_type cols, size_type default_stride, KernelFunction fn,
    KernelArgs... args)
{
    auto tidx = thread::get_thread_id_flat();
    auto col = tidx % cols;
    auto row = tidx / cols;
    if (row >= rows) {
        return;
    }
    fn(row, col,
       device_unpack_solver_impl<KernelArgs>::unpack(args, default_stride)...);
}


}  // namespace cuda
}  // namespace kernels


namespace solver {


template <typename ValueType>
kernels::cuda::default_stride_dense_wrapper<kernels::cuda::cuda_type<ValueType>>
default_stride(matrix::Dense<ValueType> *mtx)
{
    return {kernels::cuda::as_cuda_type(mtx->get_values())};
}


template <typename ValueType>
kernels::cuda::default_stride_dense_wrapper<
    const kernels::cuda::cuda_type<ValueType>>
default_stride(const matrix::Dense<ValueType> *mtx)
{
    return {kernels::cuda::as_cuda_type(mtx->get_const_values())};
}


template <typename ValueType>
kernels::cuda::cuda_type<ValueType> *row_vector(matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return kernels::cuda::as_cuda_type(mtx->get_values());
}


template <typename ValueType>
const kernels::cuda::cuda_type<ValueType> *row_vector(
    const matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return kernels::cuda::as_cuda_type(mtx->get_const_values());
}


}  // namespace solver


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const CudaExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs &&... args)
{
    cuda::device_guard guard{exec->get_device_id()};
    constexpr auto block_size = kernels::cuda::default_block_size;
    auto num_blocks = ceildiv(size[0] * size[1], block_size);
    kernels::cuda::generic_kernel_2d_solver<<<num_blocks, block_size>>>(
        size[0], size[1], default_stride, fn,
        kernels::cuda::map_to_device(args)...);
}


}  // namespace gko

#endif  // GKO_CUDA_BASE_KERNEL_LAUNCH_SOLVER_CUH_
