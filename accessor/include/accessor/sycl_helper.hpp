// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef ACCESSOR_SYCL_HELPER_HPP_
#define ACCESSOR_SYCL_HELPER_HPP_


#include <complex>
#include <type_traits>

#include "accessor/block_col_major.hpp"
#include "accessor/reduced_row_major.hpp"
#include "accessor/row_major.hpp"
#include "accessor/scaled_reduced_row_major.hpp"
#include "accessor/utils.hpp"


namespace acc {


/**
 * Maps a host type `T` to its SYCL device equivalent.
 *
 * This is the extension point for reduced-precision or otherwise
 * device-specific scalar types: specialize it for your own type, e.g.
 * @code
 * template <> struct sycl_type<my_half> { using type = sycl::half; };
 * @endcode
 * Any type without a specialization maps to itself.
 */
template <typename T>
struct sycl_type {
    using type = T;
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


/**
 * This is an alias for SYCL's equivalent of `T`.
 *
 * @tparam T  a type
 */
template <typename T>
using sycl_type_t = typename sycl_type<T>::type;


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
template <std::size_t dim, typename Type1, typename Type2, typename IndexType>
MACC_INLINE auto as_sycl_range(
    const range<reduced_row_major<dim, Type1, Type2, IndexType>>& r)
{
    return range<reduced_row_major<dim, sycl_type_t<Type1>, sycl_type_t<Type2>,
                                   IndexType>>(
        r.get_accessor().get_size(),
        as_sycl_type(r.get_accessor().get_stored_data()),
        r.get_accessor().get_stride());
}

/**
 * @copydoc as_sycl_range()
 */
template <std::size_t dim, typename Type1, typename Type2, std::uint64_t mask,
          typename IndexType>
MACC_INLINE auto as_sycl_range(
    const range<scaled_reduced_row_major<dim, Type1, Type2, mask, IndexType>>&
        r)
{
    return range<scaled_reduced_row_major<dim, sycl_type_t<Type1>,
                                          sycl_type_t<Type2>, mask, IndexType>>(
        r.get_accessor().get_size(),
        as_sycl_type(r.get_accessor().get_stored_data()),
        r.get_accessor().get_storage_stride(),
        as_sycl_type(r.get_accessor().get_scalar()),
        r.get_accessor().get_scalar_stride());
}

/**
 * @copydoc as_sycl_range()
 */
template <typename T, std::size_t dim, typename IndexType>
MACC_INLINE auto as_sycl_range(
    const range<block_col_major<T, dim, IndexType>>& r)
{
    return range<block_col_major<sycl_type_t<T>, dim, IndexType>>(
        r.get_accessor().lengths, as_sycl_type(r.get_accessor().data),
        r.get_accessor().stride);
}

/**
 * @copydoc as_sycl_range()
 */
template <typename T, size_type dim, typename IndexType>
MACC_INLINE auto as_sycl_range(const range<row_major<T, dim, IndexType>>& r)
{
    return range<block_col_major<sycl_type_t<T>, dim, IndexType>>(
        r.get_accessor().lengths, as_sycl_type(r.get_accessor().data),
        r.get_accessor().stride);
}

template <typename AccType>
MACC_INLINE auto as_device_range(AccType&& acc)
{
    return as_sycl_range(std::forward<AccType>(acc));
}


}  // namespace acc


#endif  // ACCESSOR_SYCL_HELPER_HPP_
