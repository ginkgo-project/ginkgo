// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_TYPES_HPP_
#define GKO_DPCPP_BASE_TYPES_HPP_


#include <type_traits>

#include <sycl/half_type.hpp>

#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/matrix_data.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace detail {


template <typename T>
struct sycl_type_impl {
    using type = T;
};

template <typename T>
struct sycl_type_impl<T*> {
    using type = typename sycl_type_impl<T>::type*;
};

template <typename T>
struct sycl_type_impl<T&> {
    using type = typename sycl_type_impl<T>::type&;
};

template <typename T>
struct sycl_type_impl<const T> {
    using type = const typename sycl_type_impl<T>::type;
};

template <typename T>
struct sycl_type_impl<volatile T> {
    using type = volatile typename sycl_type_impl<T>::type;
};

template <>
struct sycl_type_impl<half> {
    using type = sycl::half;
};

template <typename T>
struct sycl_type_impl<std::complex<T>> {
    using type = std::complex<typename sycl_type_impl<T>::type>;
};

template <typename ValueType, typename IndexType>
struct sycl_type_impl<matrix_data_entry<ValueType, IndexType>> {
    using type =
        matrix_data_entry<typename sycl_type_impl<ValueType>::type, IndexType>;
};

}  // namespace detail


/**
 * This is an alias for SYCL's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using sycl_type = typename detail::sycl_type_impl<T>::type;

/**
 * This is an alias for SYCL/HIP's equivalent of `T` depending on the namespace.
 *
 * @tparam T  a type
 */
template <typename T>
using device_type = sycl_type<T>;


/**
 * Reinterprets the passed in value as a SYCL type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to SYCL type
 */
template <typename T>
inline std::enable_if_t<
    std::is_pointer<T>::value || std::is_reference<T>::value, sycl_type<T>>
as_sycl_type(T val)
{
    return reinterpret_cast<sycl_type<T>>(val);
}


/**
 * @copydoc as_sycl_type()
 */
template <typename T>
inline std::enable_if_t<
    !std::is_pointer<T>::value && !std::is_reference<T>::value, sycl_type<T>>
as_sycl_type(T val)
{
    return *reinterpret_cast<sycl_type<T>*>(&val);
}


/**
 * Reinterprets the passed in value as a SYCL/HIP type depending on the
 * namespace.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to SYCL/HIP type
 */
template <typename T>
inline device_type<T> as_device_type(T val)
{
    return as_sycl_type(val);
}


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko

#endif  // GKO_DPCPP_BASE_TYPES_HPP_
