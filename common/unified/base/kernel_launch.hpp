// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
#define GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
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
using unpack_member_type = typename detail::fake_complex_unpack_impl<T>::type;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return fake_complex_unpack(value);
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
using unpack_member_type = typename detail::fake_complex_unpack_impl<T>::type;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
{
    return fake_complex_unpack(value);
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


template <typename T>
using unpack_member_type = T;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
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


template <typename T>
using unpack_member_type = T;

template <typename T>
GKO_INLINE GKO_ATTRIBUTES constexpr unpack_member_type<T> unpack_member(T value)
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


/**
 * @internal
 * A simple row-major accessor as a device representation of gko::matrix::Dense
 * objects.
 *
 * @tparam ValueType  the value type of the underlying matrix.
 */
template <typename ValueType>
struct matrix_accessor {
    ValueType* data;
    int64 stride;

    /**
     * @internal
     * Returns a reference to the element at position (row, col).
     */
    GKO_INLINE GKO_ATTRIBUTES ValueType& operator()(int64 row, int64 col)
    {
        return data[row * stride + col];
    }

    /**
     * @internal
     * Returns a reference to the element at position idx in the underlying
     * storage.
     */
    GKO_INLINE GKO_ATTRIBUTES ValueType& operator[](int64 idx)
    {
        return data[idx];
    }
};


/**
 * @internal
 * This struct is used to provide mappings from host types like
 * gko::matrix::Dense to device representations of the same data, like an
 * accessor storing only data pointer and stride.
 *
 * By default, it only maps std::complex to the corresponding device
 * representation of the complex type. There are specializations for dealing
 * with gko::array and gko::matrix::Dense (both const and mutable) that map them
 * to plain pointers or matrix_accessor objects.
 *
 * @tparam T  the type being mapped. It will be used based on a
 *            forwarding-reference, i.e. preserve references in the input
 *            parameter, so special care must be taken to only return types that
 *            can be passed to the device, i.e. (structs containing) device
 *            pointers or values. This means that T will be either a r-value or
 *            l-value reference.
 */
template <typename T>
struct to_device_type_impl {
    using type = std::decay_t<device_type<T>>;
    static type map_to_device(T in) { return as_device_type(in); }
};

template <typename ValueType>
struct to_device_type_impl<matrix::Dense<ValueType>*&> {
    using type = matrix_accessor<device_type<ValueType>>;
    static type map_to_device(matrix::Dense<ValueType>* mtx)
    {
        return to_device_type_impl<
            matrix::Dense<ValueType>* const&>::map_to_device(mtx);
    }
};

template <typename ValueType>
struct to_device_type_impl<matrix::Dense<ValueType>* const&> {
    using type = matrix_accessor<device_type<ValueType>>;
    static type map_to_device(matrix::Dense<ValueType>* mtx)
    {
        return {as_device_type(mtx->get_values()),
                static_cast<int64>(mtx->get_stride())};
    }
};

template <typename ValueType>
struct to_device_type_impl<const matrix::Dense<ValueType>*&> {
    using type = matrix_accessor<const device_type<ValueType>>;
    static type map_to_device(const matrix::Dense<ValueType>* mtx)
    {
        return {as_device_type(mtx->get_const_values()),
                static_cast<int64>(mtx->get_stride())};
    }
};

template <typename ValueType>
struct to_device_type_impl<array<ValueType>&> {
    using type = device_type<ValueType>*;
    static type map_to_device(array<ValueType>& array)
    {
        return as_device_type(array.get_data());
    }
};

template <typename ValueType>
struct to_device_type_impl<const array<ValueType>&> {
    using type = const device_type<ValueType>*;
    static type map_to_device(const array<ValueType>& array)
    {
        return as_device_type(array.get_const_data());
    }
};


template <typename T>
typename to_device_type_impl<T>::type map_to_device(T&& param)
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


#endif  // GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
