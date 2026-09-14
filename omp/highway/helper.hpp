// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_OMP_HIGHWAY_HELPER_HPP_
#define GKO_OMP_HIGHWAY_HELPER_HPP_

#include <type_traits>

namespace hwy {

struct float16_t;

struct bfloat16_t;

};  // namespace hwy


namespace gko {


class half;
class bfloat16;


namespace kernels {
namespace omp {
namespace detail {

template <typename T>
struct hwy_type_impl {
    using type = T;
};

// Unpack cv and reference / pointer qualifiers
template <typename T>
struct hwy_type_impl<const T> {
    using type = const typename hwy_type_impl<T>::type;
};

template <typename T>
struct hwy_type_impl<volatile T> {
    using type = volatile typename hwy_type_impl<T>::type;
};

template <typename T>
struct hwy_type_impl<T*> {
    using type = typename hwy_type_impl<T>::type*;
};

template <typename T>
struct hwy_type_impl<T&> {
    using type = typename hwy_type_impl<T>::type&;
};

template <typename T>
struct hwy_type_impl<T&&> {
    using type = typename hwy_type_impl<T>::type&&;
};

template <>
struct hwy_type_impl<gko::half> {
    using type = hwy::float16_t;
};

template <>
struct hwy_type_impl<gko::bfloat16> {
    using type = hwy::bfloat16_t;
};


}  // namespace detail


/**
 * This is an alias for equivalent of type T used in the Highway libary
 *
 * @tparam T  a type
 */
template <typename T>
using hwy_type = typename detail::hwy_type_impl<T>::type;

/**
 * Reinterprets the passed in value as a Highway type.
 *
 * @param val  the value to reinterpret
 *
 * @return `val` reinterpreted to Highway type
 */
template <typename T>
inline std::enable_if_t<
    std::is_pointer<T>::value || std::is_reference<T>::value, hwy_type<T>>
as_hwy_type(T val)
{
    return reinterpret_cast<hwy_type<T>>(val);
}


/**
 * @copydoc as_hwy_type()
 */
template <typename T>
inline std::enable_if_t<
    !std::is_pointer<T>::value && !std::is_reference<T>::value, hwy_type<T>>
as_hwy_type(T val)
{
    return *reinterpret_cast<hwy_type<T>*>(&val);
}


}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_HIGHWAY_HELPER_HPP_
