// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

namespace gko {


enum struct precision {
    none,
    any,
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


}  // namespace gko
