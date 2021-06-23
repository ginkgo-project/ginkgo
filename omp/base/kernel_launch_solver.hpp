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

#ifndef GKO_OMP_BASE_KERNEL_LAUNCH_SOLVER_HPP_
#define GKO_OMP_BASE_KERNEL_LAUNCH_SOLVER_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "omp/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace omp {


template <typename ValueType>
struct default_stride_dense_wrapper {
    ValueType *data;
};


template <typename T>
struct device_unpack_solver_impl {
    using type = T;
    static type unpack(T param, size_type) { return param; }
};

template <typename ValueType>
struct device_unpack_solver_impl<default_stride_dense_wrapper<ValueType>> {
    using type = matrix_accessor<ValueType>;
    static type unpack(default_stride_dense_wrapper<ValueType> param,
                       size_type default_stride)
    {
        return {param.data, default_stride};
    }
};


template <typename T>
typename device_unpack_solver_impl<typename to_device_type_impl<T>::type>::type
map_to_device_solver(T &&param, size_type default_stride)
{
    return device_unpack_solver_impl<typename to_device_type_impl<T>::type>::
        unpack(to_device_type_impl<T>::map_to_device(param), default_stride);
}


}  // namespace omp
}  // namespace kernels


namespace solver {


template <typename ValueType>
kernels::omp::default_stride_dense_wrapper<ValueType> default_stride(
    matrix::Dense<ValueType> *mtx)
{
    return {mtx->get_values()};
}


template <typename ValueType>
kernels::omp::default_stride_dense_wrapper<const ValueType> default_stride(
    const matrix::Dense<ValueType> *mtx)
{
    return {mtx->get_const_values()};
}


template <typename ValueType>
ValueType *row_vector(matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return mtx->get_values();
}


template <typename ValueType>
const ValueType *row_vector(const matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1);
    return mtx->get_const_values();
}


}  // namespace solver


template <typename KernelFunction, typename... KernelArgs>
void run_kernel_solver(std::shared_ptr<const OmpExecutor> exec,
                       KernelFunction fn, dim<2> size, size_type default_stride,
                       KernelArgs &&... args)
{
    run_kernel_impl(
        exec, fn, size,
        kernels::omp::map_to_device_solver(args, default_stride)...);
}


}  // namespace gko

#endif  // GKO_OMP_BASE_KERNEL_LAUNCH_SOLVER_HPP_
