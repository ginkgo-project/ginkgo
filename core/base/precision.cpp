// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/precision.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {


precision as_real(precision p)
{
    if (is_real(p)) {
        return p;
    }
    switch (p) {
    case precision::complex_fp32:
        return precision::fp32;
    case precision::complex_fp64:
        return precision::fp64;
#if GINKGO_ENABLE_HALF
    case precision::complex_fp16:
        return precision::fp16;
#endif
#if GINKGO_ENABLE_BFLOAT16
    case precision::complex_bf16:
        return precision::bf16;
#endif
    default:
        GKO_INVALID_STATE("Unsupported precision");
    }
}


precision as_complex(precision p)
{
    if (is_complex(p)) {
        return p;
    }
    switch (p) {
    case precision::fp32:
        return precision::complex_fp32;
    case precision::fp64:
        return precision::complex_fp64;
#if GINKGO_ENABLE_HALF
    case precision::fp16:
        return precision::complex_fp16;
#endif
#if GINKGO_ENABLE_BFLOAT16
    case precision::bf16:
        return precision::complex_bf16;
#endif
    default:
        GKO_INVALID_STATE("Unsupported precision");
    }
}


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
    case precision::any:
        return "any";
    case precision::none:
        return "none";
    default:
        GKO_INVALID_STATE("Unsupported precision");
    }
}


std::variant<
#if GINKGO_ENABLE_HALF
    half, std::complex<half>,
#endif
#if GINKGO_ENABLE_BFLOAT16
    bfloat16, std::complex<bfloat16>,
#endif
    float, std::complex<float>, double, std::complex<double>>
precision_to_variant(precision p)
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
