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

#ifndef GKO_CORE_BASE_KERNEL_LAUNCH_HPP_
#define GKO_CORE_BASE_KERNEL_LAUNCH_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#if defined(GKO_COMPILING_CUDA)

#define GKO_DEVICE_NAMESPACE cuda
#define GKO_KERNEL __device__
#include "cuda/base/types.hpp"


namespace gko {
namespace kernels {
namespace cuda {


template <typename T>
using device_type = typename detail::cuda_type_impl<T>::type;

template <typename T>
device_type<T> as_device_type(T value)
{
    return as_cuda_type(value);
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_HIP)

#define GKO_DEVICE_NAMESPACE hip
#define GKO_KERNEL __device__
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


template <typename T>
using device_type = typename detail::hip_type_impl<T>::type;

template <typename T>
device_type<T> as_device_type(T value)
{
    return as_hip_type(value);
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_DPCPP)

#define GKO_DEVICE_NAMESPACE dpcpp
#define GKO_KERNEL


namespace gko {
namespace kernels {
namespace dpcpp {


template <typename T>
using device_type = T;

template <typename T>
device_type<T> as_device_type(T value)
{
    return value;
}

}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#elif defined(GKO_COMPILING_OMP)

#define GKO_DEVICE_NAMESPACE omp
#define GKO_KERNEL


namespace gko {
namespace kernels {
namespace omp {


template <typename T>
using device_type = T;

template <typename T>
device_type<T> as_device_type(T value)
{
    return value;
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#else

#error "This file should only be used inside Ginkgo device compilation"

#endif


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {


template <typename ValueType>
struct matrix_accessor {
    ValueType *data;
    size_type stride;

    GKO_INLINE GKO_ATTRIBUTES ValueType &operator()(size_type row,
                                                    size_type col)
    {
        return data[row * stride + col];
    }

    GKO_INLINE GKO_ATTRIBUTES ValueType &operator[](size_type idx)
    {
        return data[idx];
    }
};


template <typename T>
struct to_device_type_impl {
    using type = std::decay_t<device_type<T>>;
    static type map_to_device(T in) { return as_device_type(in); }
};

template <typename ValueType>
struct to_device_type_impl<matrix::Dense<ValueType> *&> {
    using type = matrix_accessor<device_type<ValueType>>;
    static type map_to_device(matrix::Dense<ValueType> *mtx)
    {
        return {as_device_type(mtx->get_values()), mtx->get_stride()};
    }
};

template <typename ValueType>
struct to_device_type_impl<const matrix::Dense<ValueType> *&> {
    using type = matrix_accessor<const device_type<ValueType>>;
    static type map_to_device(const matrix::Dense<ValueType> *mtx)
    {
        return {as_device_type(mtx->get_const_values()), mtx->get_stride()};
    }
};

template <typename ValueType>
struct to_device_type_impl<Array<ValueType> &> {
    using type = device_type<ValueType> *;
    static type map_to_device(Array<ValueType> &array)
    {
        return as_device_type(array.get_data());
    }
};

template <typename ValueType>
struct to_device_type_impl<const Array<ValueType> &> {
    using type = const device_type<ValueType> *;
    static type map_to_device(const Array<ValueType> &array)
    {
        return as_device_type(array.get_const_data());
    }
};


template <typename T>
typename to_device_type_impl<T>::type map_to_device(T &&param)
{
    return to_device_type_impl<T>::map_to_device(param);
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/kernel_launch.cuh"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/kernel_launch.hip.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/base/kernel_launch.dp.hpp"
#elif defined(GKO_COMPILING_OMP)
#include "omp/base/kernel_launch.hpp"
#endif


#endif  // GKO_CORE_BASE_KERNEL_LAUNCH_HPP_
