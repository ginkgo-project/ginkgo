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

#ifndef GKO_OMP_BASE_KERNEL_LAUNCH_HPP_
#define GKO_OMP_BASE_KERNEL_LAUNCH_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#ifdef GKO_KERNEL
#error "Only one kernel_launch.hpp file can be included at a time."
#else
#define GKO_KERNEL
#endif


namespace gko {
namespace kernels {
namespace omp {


template <typename ValueType>
struct matrix_accessor {
    ValueType *data;
    size_type stride;

    ValueType &operator()(size_type row, size_type col)
    {
        return data[row * stride + col];
    }

    ValueType &operator[](size_type idx) { return data[idx]; }
};


template <typename ValueType>
matrix_accessor<ValueType> map_to_device(matrix::Dense<ValueType> *mtx)
{
    return {mtx->get_values(), mtx->get_stride()};
}

template <typename ValueType>
matrix_accessor<const ValueType> map_to_device(
    const matrix::Dense<ValueType> *mtx)
{
    return {mtx->get_const_values(), mtx->get_stride()};
}

template <typename ValueType>
typename std::enable_if<std::is_arithmetic<ValueType>::value, ValueType>::type *
map_to_device(ValueType *data)
{
    return data;
}

template <typename ValueType>
std::complex<ValueType> *map_to_device(std::complex<ValueType> *data)
{
    return data;
}

template <typename ValueType>
const std::complex<ValueType> *map_to_device(
    const std::complex<ValueType> *data)
{
    return data;
}

template <typename ValueType>
ValueType *map_to_device(Array<ValueType> &mtx)
{
    return mtx.get_data();
}

template <typename ValueType>
const ValueType *map_to_device(const Array<ValueType> &mtx)
{
    return mtx.get_const_data();
}


}  // namespace omp
}  // namespace kernels


template <typename KernelFunction, typename... KernelArgs>
void OmpExecutor::run_kernel(KernelFunction fn, size_type size,
                             KernelArgs &&... args) const
{
#pragma omp parallel for
    for (size_type i = 0; i < size; i++) {
        [&]() { fn(i, kernels::omp::map_to_device(args)...); }();
    }
}


template <typename KernelFunction, typename... KernelArgs>
void OmpExecutor::run_kernel(KernelFunction fn, dim<2> size,
                             KernelArgs &&... args) const
{
#pragma omp parallel for
    for (size_type i = 0; i < size[0] * size[1]; i++) {
        auto row = i / size[1];
        auto col = i % size[1];
        [&]() { fn(row, col, kernels::omp::map_to_device(args)...); }();
    }
}


}  // namespace gko

#endif  // GKO_OMP_BASE_KERNEL_LAUNCH_HPP_
