// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_HIP_HELPER_HPP_
#define GKO_ACCESSOR_HIP_HELPER_HPP_


#include <type_traits>

#include <thrust/complex.h>

#include "convert_range.hpp"
#include "range.hpp"
#include "utils.hpp"


struct __half;

struct hip_bfloat16;

struct __hip_bfloat16;


namespace gko {


class half;

class bfloat16;


namespace acc {
namespace detail {


template <typename T>
struct hip_type {
    using type = T;
};

// Unpack cv and reference / pointer qualifiers
template <typename T>
struct hip_type<const T> {
    using type = const typename hip_type<T>::type;
};

template <typename T>
struct hip_type<volatile T> {
    using type = volatile typename hip_type<T>::type;
};

template <typename T>
struct hip_type<T*> {
    using type = typename hip_type<T>::type*;
};

template <typename T>
struct hip_type<T&> {
    using type = typename hip_type<T>::type&;
};

template <typename T>
struct hip_type<T&&> {
    using type = typename hip_type<T>::type&&;
};

template <>
struct hip_type<gko::half> {
    using type = __half;
};

template <>
struct hip_type<gko::bfloat16> {
    using type = __hip_bfloat16;
};


// Transform std::complex to thrust::complex
template <typename T>
struct hip_type<std::complex<T>> {
    using type = thrust::complex<typename hip_type<T>::type>;
};


}  // namespace detail


/**
 * This is an alias for HIP's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using hip_type_t = typename detail::hip_type<T>::type;


/**
 * Reinterprets the passed in value as a HIP type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to HIP type
 */
template <typename T>
std::enable_if_t<std::is_pointer<T>::value || std::is_reference<T>::value,
                 hip_type_t<T>>
as_hip_type(T val)
{
    return reinterpret_cast<hip_type_t<T>>(val);
}


/**
 * @copydoc as_hip_type()
 */
template <typename T>
std::enable_if_t<!std::is_pointer<T>::value && !std::is_reference<T>::value,
                 hip_type_t<T>>
as_hip_type(T val)
{
    return *reinterpret_cast<hip_type_t<T>*>(&val);
}


/**
 * Changes the types and reinterprets the passed in range pointers as a HIP
 * types.
 *
 * @param r  the range which pointers need to be reinterpreted
 *
 * @return `r` with appropriate types and reinterpreted to HIP pointers
 */
template <typename Accessor>
GKO_ACC_INLINE auto as_hip_range(const range<Accessor>& r)
{
    return detail::convert_range<detail::hip_type>(r);
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_HIP_HELPER_HPP_
