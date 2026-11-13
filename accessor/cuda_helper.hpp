// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_CUDA_HELPER_HPP_
#define GKO_ACCESSOR_CUDA_HELPER_HPP_


#include <type_traits>

#include <thrust/complex.h>

#include "convert_range.hpp"
#include "range.hpp"
#include "utils.hpp"


struct __half;

struct __nv_bfloat16;


namespace gko {


class half;

class bfloat16;


namespace acc {
namespace detail {


template <typename T>
struct cuda_type {
    using type = T;
};

template <>
struct cuda_type<gko::half> {
    using type = __half;
};

template <>
struct cuda_type<gko::bfloat16> {
    using type = __nv_bfloat16;
};

// Unpack cv and reference / pointer qualifiers
template <typename T>
struct cuda_type<const T> {
    using type = const typename cuda_type<T>::type;
};

template <typename T>
struct cuda_type<volatile T> {
    using type = volatile typename cuda_type<T>::type;
};

template <typename T>
struct cuda_type<T*> {
    using type = typename cuda_type<T>::type*;
};

template <typename T>
struct cuda_type<T&> {
    using type = typename cuda_type<T>::type&;
};

template <typename T>
struct cuda_type<T&&> {
    using type = typename cuda_type<T>::type&&;
};


// Transform std::complex to thrust::complex
template <typename T>
struct cuda_type<std::complex<T>> {
    using type = thrust::complex<typename cuda_type<T>::type>;
};


}  // namespace detail


/**
 * This is an alias for CUDA's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using cuda_type_t = typename detail::cuda_type<T>::type;


/**
 * Reinterprets the passed in value as a CUDA type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to CUDA type
 */
template <typename T>
std::enable_if_t<std::is_pointer<T>::value || std::is_reference<T>::value,
                 cuda_type_t<T>>
as_cuda_type(T val)
{
    return reinterpret_cast<cuda_type_t<T>>(val);
}


/**
 * @copydoc as_cuda_type()
 */
template <typename T>
std::enable_if_t<!std::is_pointer<T>::value && !std::is_reference<T>::value,
                 cuda_type_t<T>>
as_cuda_type(T val)
{
    return *reinterpret_cast<cuda_type_t<T>*>(&val);
}


/**
 * Changes the types and reinterprets the passed in range pointers as a CUDA
 * types.
 *
 * @param r  the range which pointers need to be reinterpreted
 *
 * @return `r` with appropriate types and reinterpreted to CUDA pointers
 */
template <typename Accessor>
GKO_ACC_INLINE auto as_cuda_range(const range<Accessor>& r)
{
    return detail::convert_range<detail::cuda_type>(r);
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_CUDA_HELPER_HPP_
