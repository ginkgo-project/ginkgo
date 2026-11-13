// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_SYCL_HELPER_HPP_
#define GKO_ACCESSOR_SYCL_HELPER_HPP_


#include <complex>
#include <type_traits>

#include "convert_range.hpp"
#include "range.hpp"
#include "utils.hpp"


namespace gko {


class half;

class bfloat16;


template <typename V>
class complex;


namespace acc {
namespace detail {


template <typename T>
struct sycl_type {
    using type = T;
};

template <>
struct sycl_type<gko::half> {
    using type = sycl::half;
};

template <>
struct sycl_type<gko::bfloat16> {
    using type = sycl::ext::oneapi::bfloat16;
};

// Unpack cv and reference / pointer qualifiers
template <typename T>
struct sycl_type<const T> {
    using type = const typename sycl_type<T>::type;
};

template <typename T>
struct sycl_type<volatile T> {
    using type = volatile typename sycl_type<T>::type;
};

template <typename T>
struct sycl_type<T*> {
    using type = typename sycl_type<T>::type*;
};

template <typename T>
struct sycl_type<T&> {
    using type = typename sycl_type<T>::type&;
};

template <typename T>
struct sycl_type<T&&> {
    using type = typename sycl_type<T>::type&&;
};

// Transform the underlying type of std::complex
template <typename T>
struct sycl_type<std::complex<T>> {
    using type = std::complex<typename sycl_type<T>::type>;
};

// handle the complex in gko for half and bfloat16
template <>
struct sycl_type<std::complex<gko::half>> {
    using type = gko::complex<typename sycl_type<gko::half>::type>;
};

template <>
struct sycl_type<std::complex<gko::bfloat16>> {
    using type = gko::complex<typename sycl_type<gko::bfloat16>::type>;
};


}  // namespace detail


/**
 * This is an alias for SYCL's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using sycl_type_t = typename detail::sycl_type<T>::type;


/**
 * Reinterprets the passed in value as a SYCL type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to SYCL type
 */
template <typename T>
std::enable_if_t<std::is_pointer<T>::value || std::is_reference<T>::value,
                 sycl_type_t<T>>
as_sycl_type(T val)
{
    return reinterpret_cast<sycl_type_t<T>>(val);
}


/**
 * @copydoc as_sycl_type()
 */
template <typename T>
std::enable_if_t<!std::is_pointer<T>::value && !std::is_reference<T>::value,
                 sycl_type_t<T>>
as_sycl_type(T val)
{
    return *reinterpret_cast<sycl_type_t<T>*>(&val);
}


/**
 * Changes the types and reinterprets the passed in range pointers as a SYCL
 * types.
 *
 * @param r  the range which pointers need to be reinterpreted
 *
 * @return `r` with appropriate types and reinterpreted to SYCL pointers
 */
template <typename Accessor>
GKO_ACC_INLINE auto as_sycl_range(const range<Accessor>& r)
{
    return detail::convert_range<detail::sycl_type>(r);
}

template <typename AccType>
GKO_ACC_INLINE auto as_device_range(AccType&& acc)
{
    return as_sycl_range(std::forward<AccType>(acc));
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_SYCL_HELPER_HPP_
