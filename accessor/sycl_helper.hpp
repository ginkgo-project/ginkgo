// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_ACCESSOR_SYCL_HELPER_HPP_
#define GKO_ACCESSOR_SYCL_HELPER_HPP_


#include <complex>
#include <type_traits>

#include "block_col_major.hpp"
#include "reduced_row_major.hpp"
#include "row_major.hpp"
#include "scaled_reduced_row_major.hpp"
#include "utils.hpp"


// namespace sycl {
// inline namespace _V1 {


// class half;


// }
// }  // namespace sycl


namespace gko {


class half;


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
template <std::size_t dim, typename Type1, typename Type2>
GKO_ACC_INLINE auto as_sycl_range(
    const range<reduced_row_major<dim, Type1, Type2>>& r)
{
    return range<
        reduced_row_major<dim, sycl_type_t<Type1>, sycl_type_t<Type2>>>(
        r.get_accessor().get_size(),
        as_sycl_type(r.get_accessor().get_stored_data()),
        r.get_accessor().get_stride());
}

/**
 * @copydoc as_sycl_range()
 */
template <std::size_t dim, typename Type1, typename Type2, std::uint64_t mask>
GKO_ACC_INLINE auto as_sycl_range(
    const range<scaled_reduced_row_major<dim, Type1, Type2, mask>>& r)
{
    return range<scaled_reduced_row_major<dim, sycl_type_t<Type1>,
                                          sycl_type_t<Type2>, mask>>(
        r.get_accessor().get_size(),
        as_sycl_type(r.get_accessor().get_stored_data()),
        r.get_accessor().get_storage_stride(),
        as_sycl_type(r.get_accessor().get_scalar()),
        r.get_accessor().get_scalar_stride());
}

/**
 * @copydoc as_sycl_range()
 */
template <typename T, size_type dim>
GKO_ACC_INLINE auto as_sycl_range(const range<block_col_major<T, dim>>& r)
{
    return range<block_col_major<sycl_type_t<T>, dim>>(
        r.get_accessor().lengths, as_sycl_type(r.get_accessor().data),
        r.get_accessor().stride);
}

/**
 * @copydoc as_sycl_range()
 */
template <typename T, size_type dim>
GKO_ACC_INLINE auto as_sycl_range(const range<row_major<T, dim>>& r)
{
    return range<block_col_major<sycl_type_t<T>, dim>>(
        r.get_accessor().lengths, as_sycl_type(r.get_accessor().data),
        r.get_accessor().stride);
}

template <typename AccType>
GKO_ACC_INLINE auto as_device_range(AccType&& acc)
{
    return as_sycl_range(std::forward<AccType>(acc));
}


}  // namespace acc
}  // namespace gko


#endif  // GKO_ACCESSOR_SYCL_HELPER_HPP_
