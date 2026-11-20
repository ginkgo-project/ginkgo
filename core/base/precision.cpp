// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/precision.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace detail {


void throw_unsupported_precision(const char* file, int line, const char* func)
{
    throw InvalidStateError(file, line, func, "Unsupported precision");
}


}  // namespace detail


std::string to_string(precision p)
{
    switch (p) {
    case precision::fp32:
        return "fp32";
    case precision::complex_fp32:
        return "complex_fp32";
    case precision::fp64:
        return "fp64";
    case precision::complex_fp64:
        return "complex_fp64";
#if GINKGO_ENABLE_HALF
    case precision::fp16:
        return "fp16";
    case precision::complex_fp16:
        return "complex_fp16";
#endif
#if GINKGO_ENABLE_BFLOAT16
    case precision::bf16:
        return "bf16";
    case precision::complex_bf16:
        return "complex_bf16";
#endif
    default:
        GKO_INVALID_STATE("Unsupported precision");
    }
}


syn::variant_from_list<supported_value_types> precision_to_variant(precision p)
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
