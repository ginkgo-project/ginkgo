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

#ifndef GKO_DPCPP_BASE_KERNEL_LAUNCH_DP_HPP_
#define GKO_DPCPP_BASE_KERNEL_LAUNCH_DP_HPP_


#include <CL/sycl.hpp>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#ifdef GKO_KERNEL
#error "Only one kernel_launch.hpp file can be included at a time."
#else
#define GKO_KERNEL
#endif


namespace gko {
namespace kernels {
namespace dpcpp {


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


template <typename ValueType>
struct compact_dense_wrapper {
    ValueType *data;
};


template <typename T>
struct device_unpack_2d_impl {
    using type = T;
    static type unpack(T param, size_type, size_type, size_type, size_type)
    {
        return param;
    }
};

template <typename ValueType>
struct device_unpack_2d_impl<compact_dense_wrapper<ValueType>> {
    using type = matrix_accessor<ValueType>;
    static type unpack(compact_dense_wrapper<ValueType> param, size_type,
                       size_type, size_type, size_type num_cols)
    {
        return {param.data, num_cols};
    }
};


template <typename KernelFunction, typename... KernelArgs>
void generic_kernel_1d(sycl::handler &cgh, size_type size, KernelFunction fn,
                       KernelArgs... args)
{
    cgh.parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx_id) {
        auto idx = static_cast<size_type>(idx_id[0]);
        fn(idx, args...);
    });
}


template <typename KernelFunction, typename... KernelArgs>
void generic_kernel_2d(sycl::handler &cgh, size_type rows, size_type cols,
                       KernelFunction fn, KernelArgs... args)
{
    cgh.parallel_for(sycl::range<2>{rows, cols}, [=](sycl::id<2> idx) {
        auto row = static_cast<size_type>(idx[0]);
        auto col = static_cast<size_type>(idx[1]);
        fn(row, col,
           device_unpack_2d_impl<KernelArgs>::unpack(args, row, col, rows,
                                                     cols)...);
    });
}


}  // namespace dpcpp
}  // namespace kernels


template <typename ValueType>
kernels::dpcpp::compact_dense_wrapper<ValueType> compact(
    matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_stride() == mtx->get_size()[1]);
    return {mtx->get_values()};
}


template <typename ValueType>
kernels::dpcpp::compact_dense_wrapper<const ValueType> compact(
    const matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_stride() == mtx->get_size()[1]);
    return {mtx->get_const_values()};
}


template <typename ValueType>
ValueType *vector(matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1 ||
               (mtx->get_size()[1] == 1 && mtx->get_stride() == 1));
    return mtx->get_values();
}


template <typename ValueType>
const ValueType *vector(const matrix::Dense<ValueType> *mtx)
{
    GKO_ASSERT(mtx->get_size()[0] == 1 ||
               (mtx->get_size()[1] == 1 && mtx->get_stride() == 1));
    return mtx->get_const_values();
}


template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DpcppExecutor> exec, KernelFunction fn,
                size_type size, KernelArgs &&... args)
{
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        kernels::dpcpp::generic_kernel_1d(
            cgh, size, fn, kernels::dpcpp::map_to_device(args)...);
    });
}

template <typename KernelFunction, typename... KernelArgs>
void run_kernel(std::shared_ptr<const DpcppExecutor> exec, KernelFunction fn,
                dim<2> size, KernelArgs &&... args)
{
    exec->get_queue()->submit([&](sycl::handler &cgh) {
        kernels::dpcpp::generic_kernel_2d(
            cgh, size[0], size[1], fn, kernels::dpcpp::map_to_device(args)...);
    });
}


}  // namespace gko


#endif  // GKO_DPCPP_BASE_KERNEL_LAUNCH_DP_HPP_
