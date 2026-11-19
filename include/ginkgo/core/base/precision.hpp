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
 * An enum to specify the precision of stored data.
 */
enum class precision : std::uint16_t {
    fp32 = 1u << 0u,
    complex_fp32 = 1u << 1u,
    fp64 = 1u << 2u,
    complex_fp64 = 1u << 3u,
#if GINKGO_ENABLE_HALF
    fp16 = 1u << 4u,
    complex_fp16 = 1u << 5u,
#endif
#if GINKGO_ENABLE_BFLOAT16
    bf16 = 1u << 6u,
    complex_bf16 = 1u << 7u,
#endif
};


// Returns the string representation of a precision
std::string to_string(precision p);


constexpr bool operator==(precision a, precision b)
{
    return static_cast<std::uint16_t>(a) == static_cast<std::uint16_t>(b);
}


constexpr bool operator!=(precision a, precision b) { return !(a == b); }


namespace detail {

template <typename T>
struct precision_value;


#define GKO_INSTANTIATE_PRECISION_VALUE(_type, _value)        \
    template <>                                               \
    struct precision_value<_type> {                           \
        static constexpr precision value = precision::_value; \
    }

GKO_INSTANTIATE_PRECISION_VALUE(float, fp32);
GKO_INSTANTIATE_PRECISION_VALUE(std::complex<float>, complex_fp32);
GKO_INSTANTIATE_PRECISION_VALUE(double, fp64);
GKO_INSTANTIATE_PRECISION_VALUE(std::complex<double>, complex_fp64);
#if GINKGO_ENABLE_HALF
GKO_INSTANTIATE_PRECISION_VALUE(half, fp16);
GKO_INSTANTIATE_PRECISION_VALUE(std::complex<half>, complex_fp16);
#endif
#if GINKGO_ENABLE_BFLOAT16
GKO_INSTANTIATE_PRECISION_VALUE(bfloat16, bf16);
GKO_INSTANTIATE_PRECISION_VALUE(std::complex<bfloat16>, complex_bf16);
#endif

#undef GKO_INSTANTIATE_PRECISION_VALUE


}  // namespace detail

/**
 * Map from compile time type to runtime precision.
 *
 * @tparam T  Value type to map to a precision
 */
template <typename T>
inline constexpr precision precision_v = detail::precision_value<T>::value;


// True if the precision is complex
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


// True if the precision is real
constexpr bool is_real(precision p) { return !is_complex(p); }


namespace detail {


/**
 * Throws an InvalidStateError for an unsupported precision.
 *
 * This is not part of the constexpr functions themselves, because the
 * exception header depends on this header.
 */
[[noreturn]] void throw_unsupported_precision(const char* file, int line,
                                              const char* func);


}  // namespace detail


/**
 * Maps a precision to its corresponding real precision.
 *
 * For example as_real(complex_fp32) == fp32.
 * For a real precision this is the identity.
 *
 * @param p  precision to map to a real precision
 * @return The real precision corresponding to p
 */
constexpr precision as_real(precision p)
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
        detail::throw_unsupported_precision(__FILE__, __LINE__, __func__);
    }
}


/**
 * Maps a precision to its corresponding complex precision.
 *
 * For example as_complex(fp32) == complex_fp32.
 * For a complex precision this is the identity.
 *
 * @param p  precision to map to a complex precision
 * @return The complex precision corresponding to p
 */
constexpr precision as_complex(precision p)
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
        detail::throw_unsupported_precision(__FILE__, __LINE__, __func__);
    }
}


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
