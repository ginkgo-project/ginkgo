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


template <typename T>
struct to_device_type_impl {
    using type = std::decay_t<T>;
    static type map_to_device(T in) { return in; }
};

template <typename ValueType>
struct to_device_type_impl<matrix::Dense<ValueType> *&> {
    using type = matrix_accessor<ValueType>;
    static type map_to_device(matrix::Dense<ValueType> *mtx)
    {
        return {mtx->get_values(), mtx->get_stride()};
    }
};

template <typename ValueType>
struct to_device_type_impl<const matrix::Dense<ValueType> *&> {
    using type = matrix_accessor<const ValueType>;
    static type map_to_device(const matrix::Dense<ValueType> *mtx)
    {
        return {mtx->get_const_values(), mtx->get_stride()};
    }
};

template <typename ValueType>
struct to_device_type_impl<Array<ValueType> &> {
    using type = ValueType *;
    static type map_to_device(Array<ValueType> &array)
    {
        return array.get_data();
    }
};

template <typename ValueType>
struct to_device_type_impl<const Array<ValueType> &> {
    using type = const ValueType *;
    static type map_to_device(const Array<ValueType> &array)
    {
        return array.get_const_data();
    }
};


template <typename T>
typename to_device_type_impl<T>::type map_to_device(T &&param)
{
    return to_device_type_impl<T>::map_to_device(param);
}


}  // namespace omp
}  // namespace kernels


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs &&... args)
{
#pragma omp parallel for
    for (size_type i = 0; i < size; i++) {
        [&]() { fn(i, kernels::omp::map_to_device(args)...); }();
    }
}

template <size_type cols, typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_fixed_cols_impl(std::shared_ptr<const OmpExecutor> exec,
                                KernelFunction fn, dim<2> size,
                                MappedKernelArgs... args)
{
    const auto rows = size[0];
#pragma omp parallel for
    for (size_type row = 0; row < rows; row++) {
#pragma unroll
        for (size_type col = 0; col < cols; col++) {
            [&]() { fn(row, col, args...); }();
        }
    }
}

template <size_type remainder_cols, typename KernelFunction,
          typename... MappedKernelArgs>
void run_kernel_blocked_cols_impl(std::shared_ptr<const OmpExecutor> exec,
                                  KernelFunction fn, dim<2> size,
                                  MappedKernelArgs... args)
{
    const auto rows = size[0];
    const auto cols = size[1];
    const auto rounded_cols = cols / 4 * 4;
    GKO_ASSERT(rounded_cols + remainder_cols == cols);
#pragma omp parallel for
    for (size_type row = 0; row < rows; row++) {
        for (size_type base_col = 0; base_col < rounded_cols; base_col += 4) {
#pragma unroll
            for (size_type i = 0; i < 4; i++) {
                [&]() { fn(row, base_col + i, args...); }();
            }
        }
#pragma unroll
        for (size_type i = 0; i < remainder_cols; i++) {
            [&]() { fn(row, rounded_cols + i, args...); }();
        }
    }
}

template <typename KernelFunction, typename... MappedKernelArgs>
void run_kernel_impl(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                     dim<2> size, MappedKernelArgs... args)
{
    const auto rows = size[0];
    const auto cols = size[1];
    if (cols <= 0) {
        return;
    }
    if (cols == 1) {
        run_kernel_fixed_cols_impl<1>(exec, fn, size, args...);
        return;
    }
    if (cols == 2) {
        run_kernel_fixed_cols_impl<2>(exec, fn, size, args...);
        return;
    }
    if (cols == 3) {
        run_kernel_fixed_cols_impl<3>(exec, fn, size, args...);
        return;
    }
    if (cols == 4) {
        run_kernel_fixed_cols_impl<4>(exec, fn, size, args...);
        return;
    }
    const auto rem_cols = cols % 4;
    if (rem_cols == 0) {
        run_kernel_blocked_cols_impl<0>(exec, fn, size, args...);
        return;
    }
    if (rem_cols == 1) {
        run_kernel_blocked_cols_impl<1>(exec, fn, size, args...);
        return;
    }
    if (rem_cols == 2) {
        run_kernel_blocked_cols_impl<2>(exec, fn, size, args...);
        return;
    }
    if (rem_cols == 3) {
        run_kernel_blocked_cols_impl<3>(exec, fn, size, args...);
        return;
    }
    // should be unreachable
    GKO_ASSERT(false);
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const OmpExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs &&... args)
{
    run_kernel_impl(exec, fn, size, kernels::omp::map_to_device(args)...);
}


}  // namespace gko

#endif  // GKO_OMP_BASE_KERNEL_LAUNCH_HPP_
