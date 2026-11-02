// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include "ginkgo/core/base/math.hpp"
#include "ginkgo/core/base/types.hpp"


namespace gko {


/** Returns an unsigned type matching the size of the given float type. */
template <typename T>
struct float_to_unsigned_impl {};

template <>
struct float_to_unsigned_impl<double> {
    using type = uint64;
};

template <>
struct float_to_unsigned_impl<float> {
    using type = uint32;
};

template <>
struct float_to_unsigned_impl<half> {
    using type = uint16;
};

template <>
struct float_to_unsigned_impl<bfloat16> {
    using type = uint16;
};

template <typename T>
using float_to_unsigned_impl_t = typename float_to_unsigned_impl<T>::type;


/**
 * Type mapping from T to the type used for NaN comparisons.
 * Only different for device types.
 */
template <typename T>
struct float_to_nan_impl {
    using type = T;
};

template <typename T>
struct float_to_nan_impl<std::complex<T>> {
    using type = std::complex<typename float_to_nan_impl<T>::type>;
};

template <typename T>
using float_to_nan_impl_t = typename float_to_nan_impl<T>::type;


/**
 * Checks if a floating point number representation matches the representation
 * of the quiet NaN with value gko::nan() exactly.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES std::enable_if_t<!is_complex_s<T>::value, bool>
is_nan_exact(const T& value)
{
    using type = float_to_unsigned_impl_t<T>;
    type value_bytes{};
    type nan_bytes{};
    auto nan_value = nan<float_to_nan_impl_t<T>>();
    memcpy(&value_bytes, &value, sizeof(value));
    memcpy(&nan_bytes, &nan_value, sizeof(value));
    return value_bytes == nan_bytes;
}


/**
 * Checks if any component of the complex value matches the quiet NaN with
 * value gko::nan() exactly.
 */
template <typename T>
GKO_INLINE GKO_ATTRIBUTES std::enable_if_t<is_complex_s<T>::value, bool>
is_nan_exact(const T& value)
{
    return is_nan_exact(value.real()) || is_nan_exact(value.imag());
}


}  // namespace gko
