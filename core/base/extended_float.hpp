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
};


}  // namespace detail


/**
 * A class providing basic support for half precision floating point types.
 *
 * For now the only feature is reduced storage compared to single precision,
 * and conversions from and to single precision floating point type.
 */
class half {
public:
    GKO_ATTRIBUTES half(float32 val) noexcept
    {
#ifdef __CUDACC__
        data = __float2half_rn(val);
#else   // __CUDACC__
        data = float2half(val);
#endif  // __CUDACC__
    }

    GKO_ATTRIBUTES operator float32() const noexcept
    {
#ifdef __CUDACC__
        return __half2float(data);
#else   // __CUDACC__
        return half2float(data);
#endif  // __CUDACC__
    }

private:
    using fp16t = detail::float_traits<float16>;
    using fp32t = detail::float_traits<float32>;
    static inline uint16 float2half(float32 val) noexcept
    {
        const auto data = reinterpret_cast<const uint32 &>(val);
        if (is_inf<float32>(data)) {
            return shift_sign<float32, float16>(data) | fp16t::exponent_mask;
        } else if (is_nan<float32>(data)) {
            return shift_sign<float32, float16>(data) | fp16t::exponent_mask |
                   fp16t::significand_mask;
        } else {
            const auto tmp = shift_sign<float32, float16>(data) |
                             shift_exponent<float32, float16>(data);
            if (is_inf<float16>(tmp)) {
                return tmp;
            } else {
                return tmp | shift_significand<float32, float16>(data);
            }
        }
    }

    static inline float32 half2float(uint16 data) noexcept
    {
        if (is_inf<float16>(data)) {
            return shift_sign<float16, float32>(data) | fp32t::exponent_mask;
        } else if (is_nan<float16>(data)) {
            return shift_sign<float16, float32>(data) | fp32t::exponent_mask |
                   fp32t::significand_mask;
        } else {
            return shift_sign<float16, float32>(data) |
                   shift_exponent<float16, float32>(data) |
                   shift_significand<float16, float32>(data);
        }
    }

    template <typename T>
    static constexpr inline bool is_inf(
        typename detail::float_traits<T>::bits_type value)
    {
        using ft = detail::float_traits<T>;
        return (value & ft::exponent_mask) == ft::exponent_mask &&
               (value & ft::significand_mask) == ft::zero;
    }

    template <typename T>
    static constexpr inline bool is_nan(
        typename detail::float_traits<T>::bits_type value)
    {
        using ft = detail::float_traits<T>;
        return (value & ft::exponent_mask) == ft::exponent_mask &&
               (value & ft::significand_mask) != ft::zero;
    }

    template <typename FromType, typename ToType>
    static inline typename detail::float_traits<ToType>::bits_type shift_sign(
        typename detail::float_traits<FromType>::bits_type value)
    {
        using fft = detail::float_traits<FromType>;
        using tft = detail::float_traits<ToType>;
        using fbt = typename fft::bits_type;
        using tbt = typename tft::bits_type;
        constexpr int exponent_offset =
            fft::significand_bits - tft::significand_bits;
        constexpr int sign_offset =
            fft::exponent_bits - tft::exponent_bits + exponent_offset;
        if (sign_offset >= 0) {
            return static_cast<tbt>((value & fft::sign_mask) >> sign_offset);
        } else {
            return static_cast<tbt>(value & fft::sign_mask) << -sign_offset;
        }
    }

    template <typename FromType, typename ToType>
    static inline typename std::enable_if<
        (sizeof(FromType) > sizeof(ToType)),
        typename detail::float_traits<ToType>::bits_type>::type
    shift_exponent(typename detail::float_traits<FromType>::bits_type value)
    {
        // downcasting larger to smaller type
        using fft = detail::float_traits<FromType>;
        using tft = detail::float_traits<ToType>;
        using fbt = typename fft::bits_type;
        using tbt = typename tft::bits_type;
        constexpr int exponent_offset =
            fft::significand_bits - tft::significand_bits;
        const auto shifted_exp =
            (value & fft::exponent_mask) >> exponent_offset;
        constexpr auto bias_change = (fft::bias_mask >> exponent_offset) -
                                     static_cast<fbt>(tft::bias_mask);
        if (bias_change >= shifted_exp) {
            return tbt{};
        } else if (shifted_exp - bias_change >=
                   static_cast<fbt>(tft::exponent_mask)) {
            return tft::exponent_mask;
        } else {
            return static_cast<tbt>(shifted_exp - bias_change);
        }
    }

    template <typename FromType, typename ToType>
    static inline typename std::enable_if<
        (sizeof(FromType) <= sizeof(ToType)),
        typename detail::float_traits<ToType>::bits_type>::type
    shift_exponent(typename detail::float_traits<FromType>::bits_type value)
    {
        // upcasing smaller to larger type
        using fft = detail::float_traits<FromType>;
        using tft = detail::float_traits<ToType>;
        using fbt = typename fft::bits_type;
        using tbt = typename tft::bits_type;
        constexpr int exponent_offset =
            tft::significand_bits - fft::significand_bits;
        const auto shifted_exp = static_cast<tbt>(value & fft::exponent_mask)
                                 << exponent_offset;
        constexpr auto bias_change =
            tft::bias_mask -
            (static_cast<tbt>(fft::bias_mask) << exponent_offset);
        if (shifted_exp == tft::zero) {
            return tbt{};
        } else {
            return shifted_exp + bias_change;
        }
    }

    template <typename FromType, typename ToType>
    static inline typename detail::float_traits<ToType>::bits_type
    shift_significand(typename detail::float_traits<FromType>::bits_type value)
    {
        using fft = detail::float_traits<FromType>;
        using tft = detail::float_traits<ToType>;
        using fbt = typename fft::bits_type;
        using tbt = typename tft::bits_type;
        constexpr int significand_offset =
            fft::significand_bits - tft::significand_bits;
        if (significand_offset >= 0) {
            return static_cast<tbt>((value & fft::significand_mask) >>
                                    significand_offset);
        } else {
            return static_cast<tbt>(value & fft::significand_mask)
                   << -significand_offset;
        }
    }

    uint16 data;
};


}  // namespace gko


#endif  // GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
