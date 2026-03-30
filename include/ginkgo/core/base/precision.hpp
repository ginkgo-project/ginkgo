// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

namespace gko {


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


// Equality comparison operator.
// The precision::none is not equal to any other precision, including none.
constexpr bool operator==(precision a, precision b)
{
    auto int_a = static_cast<int>(a);
    auto int_b = static_cast<int>(b);
    if (int_a == static_cast<int>(precision::any) ||
        int_b == static_cast<int>(precision::any)) {
        return true;
    }
    if (int_a == static_cast<int>(precision::none) ||
        int_b == static_cast<int>(precision::none)) {
        return false;
    }
    return int_a == int_b;
}


constexpr bool operator!=(precision a, precision b) { return !(a == b); }


template <typename T>
inline precision type_to_precision;

template <>
inline constexpr precision type_to_precision<float> = precision::fp32;
template <>
inline constexpr precision type_to_precision<std::complex<float>> =
    precision::complex_fp32;
template <>
inline constexpr precision type_to_precision<double> = precision::fp64;
template <>
inline constexpr precision type_to_precision<std::complex<double>> =
    precision::complex_fp64;
#if GINKGO_ENABLE_HALF
template <>
inline constexpr precision type_to_precision<half> = precision::fp16;
template <>
inline constexpr precision type_to_precision<std::complex<half>> =
    precision::complex_fp16;
#endif
#if GINKGO_ENABLE_BFLOAT16
template <>
inline constexpr precision type_to_precision<bfloat16> = precision::bf16;
template <>
inline constexpr precision type_to_precision<std::complex<bfloat16>> =
    precision::complex_bf16;
#endif


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


inline auto precision_to_variant(precision p) -> std::variant<
#if GINKGO_ENABLE_HALF
    half, std::complex<half>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    bfloat16, std::complex<bfloat16>,
#endif
    float, std::complex<float>, double, std::complex<double>>
{
    switch (p) {
#if GINKGO_ENABLE_HALF
    case precision::fp16:
        return half{};
    case precision::complex_fp16:
        return std::complex<half>{};
#endif
#if GINKGO_ENABLE_BFLOAT16
    case precision::bf16:
        return bfloat16{};
    case precision::complex_bf16:
        return std::complex<bfloat16>{};
#endif
    case precision::fp32:
        return float{};
    case precision::complex_fp32:
        return std::complex<float>{};
    case precision::fp64:
        return double{};
    case precision::complex_fp64:
        return std::complex<double>{};
    default:
        GKO_INVALID_STATE("Unsupported precision");
    }
}


}  // namespace gko
