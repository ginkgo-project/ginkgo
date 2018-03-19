/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
#define GKO_CORE_BASE_EXTENDED_FLOAT_HPP_


#include "core/base/types.hpp"


namespace gko {
namespace detail {


template <typename T>
struct basic_float_traits;

template <>
struct basic_float_traits<float16> {
    using type = float16;
    using bits_type = uint16;
    static constexpr int significand_bits = 10;
    static constexpr int exponent_bits = 5;
};

template <>
struct basic_float_traits<float32> {
    using type = float32;
    using bits_type = uint32;
    static constexpr int significand_bits = 23;
    static constexpr int exponent_bits = 8;
};

template <>
struct basic_float_traits<float64> {
    using type = float64;
    using bits_type = uint64;
    static constexpr int significand_bits = 52;
    static constexpr int exponent_bits = 11;
};


template <typename T>
struct float_traits {
    using type = typename basic_float_traits<T>::type;
    using bits_type = typename basic_float_traits<T>::bits_type;
    static constexpr int significand_bits =
        basic_float_traits<T>::significand_bits;
    static constexpr int exponent_bits = basic_float_traits<T>::exponent_bits;
    static constexpr bits_type zero = 0;
    static constexpr bits_type one = 1;
    static constexpr bits_type significand_mask =
        (one << significand_bits) - one;
    static constexpr bits_type exponent_mask =
        (one << significand_bits + exponent_bits) - one - significand_mask;
    static constexpr bits_type bias_mask =
        (one << significand_bits + exponent_bits - 1) - one - significand_mask;
    static constexpr bits_type sign_mask = one
                                           << significand_bits + exponent_bits;

    static constexpr bool is_inf(bits_type data)
    {
        return (data & exponent_mask) == exponent_mask &&
               (data & significand_mask) == zero;
    }

    static constexpr bool is_nan(bits_type data)
    {
        return (data & exponent_mask) == exponent_mask &&
               (data & significand_mask) != zero;
    }

    static constexpr bool is_denom(bits_type data)
    {
        return (data & exponent_mask) == zero;
    }
};


template <typename SourceType, typename ResultType,
          bool = (sizeof(SourceType) <= sizeof(ResultType))>
struct precision_converter;

// upcasting implementation details
template <typename SourceType, typename ResultType>
struct precision_converter<SourceType, ResultType, true> {
    using source_traits = float_traits<SourceType>;
    using result_traits = float_traits<ResultType>;
    using source_bits = typename source_traits::bits_type;
    using result_bits = typename result_traits::bits_type;

    static_assert(source_traits::exponent_bits <=
                          result_traits::exponent_bits &&
                      source_traits::significand_bits <=
                          result_traits::significand_bits,
                  "SourceType has to have both lower range and precision or "
                  "higher range and precision than ResultType");

    static constexpr int significand_offset =
        result_traits::significand_bits - source_traits::significand_bits;
    static constexpr int exponent_offset = significand_offset;
    static constexpr int sign_offset = result_traits::exponent_bits -
                                       source_traits::exponent_bits +
                                       exponent_offset;
    static constexpr result_bits bias_change =
        result_traits::bias_mask -
        (static_cast<result_bits>(source_traits::bias_mask) << exponent_offset);

    static constexpr result_bits shift_significand(source_bits data) noexcept
    {
        return static_cast<result_bits>(data & source_traits::significand_mask)
               << significand_offset;
    }

    static constexpr result_bits shift_exponent(source_bits data) noexcept
    {
        return update_bias(
            static_cast<result_bits>(data & source_traits::exponent_mask)
            << exponent_offset);
    }

    static constexpr result_bits shift_sign(source_bits data) noexcept
    {
        return static_cast<result_bits>(data & source_traits::sign_mask)
               << sign_offset;
    }

private:
    static constexpr result_bits update_bias(result_bits data) noexcept
    {
        return data == result_traits::zero ? data : data + bias_change;
    }
};

// downcasting implementation details
template <typename SourceType, typename ResultType>
struct precision_converter<SourceType, ResultType, false> {
    using source_traits = float_traits<SourceType>;
    using result_traits = float_traits<ResultType>;
    using source_bits = typename source_traits::bits_type;
    using result_bits = typename result_traits::bits_type;

    static_assert(source_traits::exponent_bits >=
                          result_traits::exponent_bits &&
                      source_traits::significand_bits >=
                          result_traits::significand_bits,
                  "SourceType has to have both lower range and precision or "
                  "higher range and precision than ResultType");

    static constexpr int significand_offset =
        source_traits::significand_bits - result_traits::significand_bits;
    static constexpr int exponent_offset = significand_offset;
    static constexpr int sign_offset = source_traits::exponent_bits -
                                       result_traits::exponent_bits +
                                       exponent_offset;
    static constexpr source_bits bias_change =
        (source_traits::bias_mask >> exponent_offset) -
        static_cast<source_bits>(result_traits::bias_mask);

    static constexpr result_bits shift_significand(source_bits data) noexcept
    {
        return static_cast<result_bits>(
            (data & source_traits::significand_mask) >> significand_offset);
    }

    static constexpr result_bits shift_exponent(source_bits data) noexcept
    {
        return static_cast<result_bits>(update_bias(
            (data & source_traits::exponent_mask) >> exponent_offset));
    }

    static constexpr result_bits shift_sign(source_bits data) noexcept
    {
        return static_cast<result_bits>((data & source_traits::sign_mask) >>
                                        sign_offset);
    }

private:
    static constexpr source_bits update_bias(source_bits data) noexcept
    {
        return data <= bias_change ? source_traits::zero
                                   : limit_exponent(data - bias_change);
    }

    static constexpr source_bits limit_exponent(source_bits data) noexcept
    {
        return data >= static_cast<source_bits>(result_traits::exponent_mask)
                   ? static_cast<source_bits>(result_traits::exponent_mask)
                   : data;
    }
};


}  // namespace detail


/**
 * A class providing basic support for half precision floating point types.
 *
 * For now the only features are reduced storage compared to single precision
 * and conversions from and to single precision floating point type.
 */
class half {
public:
    GKO_ATTRIBUTES half() = default;

    GKO_ATTRIBUTES half(float32 val) noexcept
    {
#ifdef __CUDACC__
        data = __float2half_rn(val);
#else   // __CUDACC__
        data = float2half(reinterpret_cast<const uint32 &>(val));
#endif  // __CUDACC__
    }

    GKO_ATTRIBUTES half(float64 val) noexcept : half(static_cast<float32>(val))
    {}

    GKO_ATTRIBUTES operator float32() const noexcept
    {
#ifdef __CUDACC__
        return __half2float(data);
#else   // __CUDACC__
        const auto bits = half2float(data);
        return reinterpret_cast<const float32 &>(bits);
#endif  // __CUDACC__
    }

    GKO_ATTRIBUTES operator float64() const noexcept
    {
        return static_cast<float64>(static_cast<float32>(*this));
    }

private:
    using f16_traits = detail::float_traits<float16>;
    using f32_traits = detail::float_traits<float32>;

    static uint16 float2half(uint32 data) noexcept
    {
        using conv = detail::precision_converter<float32, float16>;
        if (f32_traits::is_inf(data)) {
            return conv::shift_sign(data) | f16_traits::exponent_mask;
        } else if (f32_traits::is_nan(data)) {
            return conv::shift_sign(data) | f16_traits::exponent_mask |
                   f16_traits::significand_mask;
        } else {
            const auto exp = conv::shift_exponent(data);
            if (f16_traits::is_inf(exp)) {
                return conv::shift_sign(data) | exp;
            } else if (f16_traits::is_denom(exp)) {
                // TODO: handle denormals
                return conv::shift_sign(data);
            } else {
                return conv::shift_sign(data) | exp |
                       conv::shift_significand(data);
            }
        }
    }

    static uint32 half2float(uint16 data) noexcept
    {
        using conv = detail::precision_converter<float16, float32>;
        if (f16_traits::is_inf(data)) {
            return conv::shift_sign(data) | f32_traits::exponent_mask;
        } else if (f16_traits::is_nan(data)) {
            return conv::shift_sign(data) | f32_traits::exponent_mask |
                   f32_traits::significand_mask;
        } else if (f16_traits::is_denom(data)) {
            // TODO: handle denormals
            return conv::shift_sign(data);
        } else {
            return conv::shift_sign(data) | conv::shift_exponent(data) |
                   conv::shift_significand(data);
        }
    }

    uint16 data;
};


}  // namespace gko


namespace std {


template <>
class complex<gko::half> {
public:
    complex(const gko::half &real = 0.f, const gko::half &imag = 0.f)
        : real_(real), imag_(imag)
    {}

    template <typename T>
    explicit complex(const complex<T> &other)
        : complex(other.real(), other.imag())
    {}

    gko::half real() const noexcept { return real_; }

    gko::half imag() const noexcept { return imag_; }


    operator std::complex<gko::float32>() const noexcept
    {
        return std::complex<gko::float32>(static_cast<gko::float32>(real_),
                                          static_cast<gko::float32>(imag_));
    }

private:
    gko::half real_;
    gko::half imag_;
};


}  // namespace std


#endif  // GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
