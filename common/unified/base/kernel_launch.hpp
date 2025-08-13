// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
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

#define GKO_KERNEL __device__
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"


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

#define GKO_KERNEL __device__
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"


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

#define GKO_KERNEL


#include "dpcpp/base/math.hpp"
#include "dpcpp/base/types.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {


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


template <typename T>
struct restricted_ptr {
    T* GKO_RESTRICT data;

    /**
     * @internal
     * Returns a reference to the element at position idx in the underlying
     * storage.
     */
    GKO_INLINE GKO_ATTRIBUTES std::add_lvalue_reference_t<T> operator[](
        int64 idx) const
    {
        return data[idx];
    }

    GKO_INLINE GKO_ATTRIBUTES std::add_lvalue_reference_t<T> operator*() const
    {
        return *data;
    }
};


template <typename T>
using aliased_ptr = T*;


/**
 * @internal
 * A simple row-major accessor as a device representation of gko::matrix::Dense
 * objects.
 *
 * @tparam ValueType  the value type of the underlying matrix.
 * @tparam PtrWrapper  the pointer type. By default, it's just `T*`, but it may
 *                     be set to restricted_ptr.
 */
template <typename ValueType,
          template <typename> typename PtrWrapper = aliased_ptr>
struct matrix_accessor {
    PtrWrapper<ValueType> data;
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
 * Tag to signal that pointers should be annotated with `__restrict`
 */
struct restrict_tag {};


/**
 * @internal
 * Adds a restrict annotation to an object.
 *
 * @note Can't be used for run_kernel_solver.
 *
 * @tparam T  Type that should be annotated
 *
 * @param orig  Original object
 *
 * @return The original object and a restrict_tag
 */
template <typename T>
auto as_restrict(T&& orig) -> std::pair<T&&, restrict_tag>
{
    return {std::forward<T>(orig), restrict_tag{}};
}


/**
 * @internal
 * This struct is used to provide mappings from host types like
 * gko::matrix::Dense to device representations of the same data, like an
 * accessor storing only data pointer and stride.
 *
 * By default, it only maps std::complex to the corresponding device
 * representation of the complex type. There are specializations for dealing
 * with gko::array and gko::matrix::Dense that map them
 * to plain pointers or matrix_accessor objects.
 *
 * @tparam T  the underlying type being mapped. Any references or const
 *            qualifiers have to be resolved before passing the type.
 *            The distinction between const/mutable objects is done by
 *            overloading the map_to_device function.
 * @tparam PtrWrapper  the pointer type. By default, it's just `T*`, but it may
 *                     be set to restricted_ptr.
 */
template <typename T, template <typename> typename PtrWrapper = aliased_ptr>
struct to_device_type_impl {
    static auto map_to_device(T in) -> device_type<T>
    {
        return as_device_type(in);
    }
};

template <typename T, template <typename> typename PtrWrapper>
struct to_device_type_impl<T*, PtrWrapper> {
    static auto map_to_device(T* in) -> PtrWrapper<device_type<T>>
    {
        return {as_device_type(in)};
    }
    static auto map_to_device(const T* in) -> PtrWrapper<const device_type<T>>
    {
        return {as_device_type(in)};
    }
};

template <typename ValueType, template <typename> typename PtrWrapper>
struct to_device_type_impl<matrix::Dense<ValueType>*, PtrWrapper> {
    static auto map_to_device(matrix::Dense<ValueType>* mtx)
        -> matrix_accessor<device_type<ValueType>, PtrWrapper>
    {
        return {as_device_type(mtx->get_values()),
                static_cast<int64>(mtx->get_stride())};
    }

    static auto map_to_device(const matrix::Dense<ValueType>* mtx)
        -> matrix_accessor<const device_type<ValueType>, PtrWrapper>
    {
        return {as_device_type(mtx->get_const_values()),
                static_cast<int64>(mtx->get_stride())};
    }
};

template <typename ValueType, template <typename> typename PtrWrapper>
struct to_device_type_impl<array<ValueType>, PtrWrapper> {
    static auto map_to_device(array<ValueType>& array)
        -> PtrWrapper<device_type<ValueType>>
    {
        return {as_device_type(array.get_data())};
    }

    static auto map_to_device(const array<ValueType>& array)
        -> PtrWrapper<const device_type<ValueType>>
    {
        return {as_device_type(array.get_const_data())};
    }
};

/**
 * Specialization for handling objects annotated by as_restrict.
 * It changes the pointer wrapper type to restricted_ptr.
 */
template <typename T>
struct to_device_type_impl<std::pair<T, restrict_tag>, aliased_ptr> {
    template <typename U>
    static auto map_to_device(U&& in)
    {
        return to_device_type_impl<T, restricted_ptr>::map_to_device(in.first);
    }
};


namespace detail {


/**
 * Similar to std::remove_cv_t except that it remove the const from pointers,
 * i.e. `const T*` -> `T*`.
 */
template <typename T>
struct aggressive_remove_const {
    using type = std::remove_cv_t<T>;
};

template <typename T>
struct aggressive_remove_const<const T*> {
    using type = T*;
};

/**
 * Similar to std::decay, except that it also applies std::decay on the first
 * nesting of types, i.e. `T<U&>` -> `T<U>`.
 * This only resolves a single level of nesting.
 */
template <typename T>
struct nested_decay;

/**
 * Helper type for nested_decay.
 * This is necessary, since the references in the top-level type have to be
 * removed, before std::decay may be applied to the nested type.
 */
template <typename T>
struct nested_decay_inner {
    using type = typename aggressive_remove_const<std::decay_t<T>>::type;
};

template <typename T, typename Tag>
struct nested_decay_inner<std::pair<T, Tag>> {
    using type = std::pair<typename nested_decay_inner<T>::type, Tag>;
};

template <typename T>
struct nested_decay {
    using type = typename nested_decay_inner<std::decay_t<T>>::type;
};


}  // namespace detail


template <typename T>
using to_device_type =
    to_device_type_impl<typename detail::nested_decay<T>::type>;

template <typename T>
auto map_to_device(T&& param)
{
    return to_device_type<T>::map_to_device(std::forward<T>(param));
}


}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
#include "common/cuda_hip/base/kernel_launch.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/base/kernel_launch.dp.hpp"
#elif defined(GKO_COMPILING_OMP)
#include "omp/base/kernel_launch.hpp"
#endif


#endif  // GKO_COMMON_UNIFIED_BASE_KERNEL_LAUNCH_HPP_
