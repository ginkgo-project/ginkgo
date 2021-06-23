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

#ifndef GKO_COMMON_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#define GKO_COMMON_BASE_KERNEL_LAUNCH_SOLVER_HPP_


#include "common/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


template <typename ValueType>
struct default_stride_dense_wrapper {
    ValueType *data;
};


template <typename T>
struct device_unpack_solver_impl {
    using type = T;
    static GKO_INLINE GKO_ATTRIBUTES type unpack(T param, size_type)
    {
        return param;
    }
};

template <typename ValueType>
struct device_unpack_solver_impl<default_stride_dense_wrapper<ValueType>> {
    using type = matrix_accessor<ValueType>;
    static GKO_INLINE GKO_ATTRIBUTES type unpack(
        default_stride_dense_wrapper<ValueType> param, size_type default_stride)
    {
        return {param.data, default_stride};
    }
};


template <typename ValueType>
default_stride_dense_wrapper<device_type<ValueType>> default_stride(
    matrix::Dense<ValueType> *mtx)
{
    return {as_device_type(mtx->get_values())};
}


template <typename ValueType>
default_stride_dense_wrapper<const device_type<ValueType>> default_stride(
    const matrix::Dense<ValueType> *mtx)
{
    return {as_device_type(mtx->get_const_values())};
}


template <typename ValueType>
device_type<ValueType> *row_vector(matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return as_device_type(mtx->get_values());
}


template <typename ValueType>
const device_type<ValueType> *row_vector(const matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return as_device_type(mtx->get_const_values());
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/kernel_launch_solver.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/kernel_launch_solver.hip.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/base/kernel_launch_solver.dp.hpp"
#elif defined(GKO_COMPILING_OMP)
#include "omp/base/kernel_launch_solver.hpp"
#endif


#endif  // GKO_COMMON_BASE_KERNEL_LAUNCH_SOLVER_HPP_
