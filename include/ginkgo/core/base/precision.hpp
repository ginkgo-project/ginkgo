// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <complex>
#include <variant>

#include <ginkgo/config.hpp>
#include <ginkgo/core/base/types.hpp>

namespace gko {


/**
 * A enum to specify the precision of stored data.
 */
enum struct precision {
    none,  //!< no precision information is available, incompatible with all
           //!< other precisions
    any,   //!< compatible with all other precisions, including none
    fp32,
    complex_fp32,
    fp64,
    complex_fp64,
#if GINKGO_ENABLE_HALF
    fp16,
    complex_fp16,
#endif
#if GINKGO_ENABLE_BFLOAT16
    bf16,
    complex_bf16,
#endif
};


// Returns the string representation of a precision
std::string to_string(precision p);


// Equality comparison operator.
// The precision::any is equal to any other precision, including none.
constexpr bool operator==(precision a, precision b)
{
    auto int_a = static_cast<int>(a);
    auto int_b = static_cast<int>(b);
    if (int_a == static_cast<int>(precision::any) ||
        int_b == static_cast<int>(precision::any)) {
        return true;
    }
    return int_a == int_b;
}


constexpr bool operator!=(precision a, precision b) { return !(a == b); }


/**
 * Map from compile time type to runtime precision.
 *
 * @tparam T  Value type to map to a precision
 */
template <typename T>
inline precision precision_v;

template <>
inline constexpr precision precision_v<float> = precision::fp32;
template <>
inline constexpr precision precision_v<std::complex<float>> =
    precision::complex_fp32;
template <>
inline constexpr precision precision_v<double> = precision::fp64;
template <>
inline constexpr precision precision_v<std::complex<double>> =
    precision::complex_fp64;
#if GINKGO_ENABLE_HALF
template <>
inline constexpr precision precision_v<half> = precision::fp16;
template <>
inline constexpr precision precision_v<std::complex<half>> =
    precision::complex_fp16;
#endif
#if GINKGO_ENABLE_BFLOAT16
template <>
inline constexpr precision precision_v<bfloat16> = precision::bf16;
template <>
inline constexpr precision precision_v<std::complex<bfloat16>> =
    precision::complex_bf16;
#endif


// True if the precision is complex or any
constexpr bool is_complex(precision p)
{
    return
#if GINKGO_ENABLE_HALF
        p == precision::complex_fp16 ||
#endif
#if GINKGO_ENABLE_BFLOAT16
        p == precision::complex_bf16 ||
#endif
        p == precision::complex_fp32 || p == precision::complex_fp64;
}


// True if the precision is real or any
constexpr bool is_real(precision p)
{
    return
#if GINKGO_ENABLE_HALF
        p == precision::fp16 ||
#endif
#if GINKGO_ENABLE_BFLOAT16
        p == precision::bf16 ||
#endif
        p == precision::fp32 || p == precision::fp64;
}


/**
 * Maps a precision to its corresponding real precision.
 *
 * For example as_real(complex_fp32) == fp32.
 * For a real precision or any this is the identity.
 *
 * @throws InvalidStateError if the precision is none
 *
 * @param p  precision to map to a real precision
 * @return The real precision corresponding to p
 */
precision as_real(precision p);


/**
 * Maps a precision to its corresponding complex precision.
 *
 * For example as_complex(fp32) == complex_fp32.
 * For a complex precision or any this is the identity.
 *
 * @throws InvalidStateError if the precision is none
 *
 * @param p  precision to map to a complex precision
 * @return The complex precision corresponding to p
 */
precision as_complex(precision p);


/**
 * Create a variant from a precision.
 *
 * This allows to map the runtime precision back to a compile time precision
 * when needed.
 * For example, for a precision fp32, the result variant will hold float as
 * its alternatives.
 *
 * @param p The precision to map to the variant
 * @return A variant which value corresponds to the type matching the precision.
 */
std::variant<
#if GINKGO_ENABLE_HALF
    half, std::complex<half>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    bfloat16, std::complex<bfloat16>,
#endif
    float, std::complex<float>, double, std::complex<double>>
precision_to_variant(precision p);


}  // namespace gko
