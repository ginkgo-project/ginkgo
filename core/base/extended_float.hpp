/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
#define GKO_CORE_BASE_EXTENDED_FLOAT_HPP_


#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>


#ifdef __CUDA_ARCH__


#include <cuda_fp16.h>


#elif defined(__HIP_DEVICE_COMPILE__)


#include <hip/hip_fp16.h>


#endif  // __CUDA_ARCH__


namespace gko {


template <typename, size_type, size_type>
class truncated;


namespace detail {


template <std::size_t, typename = void>
struct uint_of_impl {};

template <std::size_t Bits>
struct uint_of_impl<Bits, xstd::void_t<xstd::enable_if_t<(Bits <= 16)>>> {
    using type = uint16;
};

template <std::size_t Bits>
struct uint_of_impl<
    Bits, xstd::void_t<xstd::enable_if_t<(16 < Bits && Bits <= 32)>>> {
    using type = uint32;
};

template <std::size_t Bits>
struct uint_of_impl<Bits, xstd::void_t<xstd::enable_if_t<(32 < Bits)>>> {
    using type = uint64;
};

template <std::size_t Bits>
using uint_of = typename uint_of_impl<Bits>::type;


template <typename T>
struct basic_float_traits {};

template <>
struct basic_float_traits<float16> {
    using type = float16;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 10;
    static constexpr int exponent_bits = 5;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<float32> {
    using type = float32;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 23;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<float64> {
    using type = float64;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 52;
    static constexpr int exponent_bits = 11;
    static constexpr bool rounds_to_nearest = true;
};

template <typename FloatType, size_type NumComponents, size_type ComponentId>
struct basic_float_traits<truncated<FloatType, NumComponents, ComponentId>> {
    using type = truncated<FloatType, NumComponents, ComponentId>;
    static constexpr int sign_bits = ComponentId == 0 ? 1 : 0;
    static constexpr int exponent_bits =
        ComponentId == 0 ? basic_float_traits<FloatType>::exponent_bits : 0;
    static constexpr int significand_bits =
        ComponentId == 0 ? sizeof(type) * byte_size - exponent_bits - 1
                         : sizeof(type) * byte_size;
    static constexpr bool rounds_to_nearest = false;
};


template <typename UintType>
constexpr UintType create_ones(int n)
{
    return (n == sizeof(UintType) * byte_size ? static_cast<UintType>(0)
                                              : static_cast<UintType>(1) << n) -
           static_cast<UintType>(1);
}

template <typename T>
struct float_traits {
    using type = typename basic_float_traits<T>::type;
    using bits_type = uint_of<sizeof(type) * byte_size>;
    static constexpr int sign_bits = basic_float_traits<T>::sign_bits;
    static constexpr int significand_bits =
        basic_float_traits<T>::significand_bits;
    static constexpr int exponent_bits = basic_float_traits<T>::exponent_bits;
    static constexpr bits_type significand_mask =
        create_ones<bits_type>(significand_bits);
    static constexpr bits_type exponent_mask =
        create_ones<bits_type>(significand_bits + exponent_bits) -
        significand_mask;
    static constexpr bits_type bias_mask =
        create_ones<bits_type>(significand_bits + exponent_bits - 1) -
        significand_mask;
    static constexpr bits_type sign_mask =
        create_ones<bits_type>(sign_bits + significand_bits + exponent_bits) -
        exponent_mask - significand_mask;
    static constexpr bool rounds_to_nearest =
        basic_float_traits<T>::rounds_to_nearest;

    static constexpr auto eps =
        1.0 / (1ll << (significand_bits + rounds_to_nearest));

    static constexpr bool is_inf(bits_type data)
    {
        return (data & exponent_mask) == exponent_mask &&
               (data & significand_mask) == bits_type{};
    }

    static constexpr bool is_nan(bits_type data)
    {
        return (data & exponent_mask) == exponent_mask &&
               (data & significand_mask) != bits_type{};
    }

    static constexpr bool is_denom(bits_type data)
    {
        return (data & exponent_mask) == bits_type{};
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
        return data == typename result_traits::bits_type{} ? data
                                                           : data + bias_change;
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
        return data <= bias_change ? typename source_traits::bits_type{}
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
    GKO_ATTRIBUTES half() noexcept = default;

    GKO_ATTRIBUTES half(float32 val) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const auto tmp = __float2half_rn(val);
        data_ = reinterpret_cast<const uint16 &>(tmp);
#else   // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        data_ = float2half(reinterpret_cast<const uint32 &>(val));
#endif  // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }

    GKO_ATTRIBUTES half(float64 val) noexcept : half(static_cast<float32>(val))
    {}

    GKO_ATTRIBUTES operator float32() const noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __half2float(reinterpret_cast<const __half &>(data_));
#else   // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const auto bits = half2float(data_);
        return reinterpret_cast<const float32 &>(bits);
#endif  // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }

    GKO_ATTRIBUTES operator float64() const noexcept
    {
        return static_cast<float64>(static_cast<float32>(*this));
    }

private:
    using f16_traits = detail::float_traits<float16>;
    using f32_traits = detail::float_traits<float32>;

    static uint16 float2half(uint32 data_) noexcept
    {
        using conv = detail::precision_converter<float32, float16>;
        if (f32_traits::is_inf(data_)) {
            return conv::shift_sign(data_) | f16_traits::exponent_mask;
        } else if (f32_traits::is_nan(data_)) {
            return conv::shift_sign(data_) | f16_traits::exponent_mask |
                   f16_traits::significand_mask;
        } else {
            const auto exp = conv::shift_exponent(data_);
            if (f16_traits::is_inf(exp)) {
                return conv::shift_sign(data_) | exp;
            } else if (f16_traits::is_denom(exp)) {
                // TODO: handle denormals
                return conv::shift_sign(data_);
            } else {
                return conv::shift_sign(data_) | exp |
                       conv::shift_significand(data_);
            }
        }
    }

    static uint32 half2float(uint16 data_) noexcept
    {
        using conv = detail::precision_converter<float16, float32>;
        if (f16_traits::is_inf(data_)) {
            return conv::shift_sign(data_) | f32_traits::exponent_mask;
        } else if (f16_traits::is_nan(data_)) {
            return conv::shift_sign(data_) | f32_traits::exponent_mask |
                   f32_traits::significand_mask;
        } else if (f16_traits::is_denom(data_)) {
            // TODO: handle denormals
            return conv::shift_sign(data_);
        } else {
            return conv::shift_sign(data_) | conv::shift_exponent(data_) |
                   conv::shift_significand(data_);
        }
    }

    uint16 data_;
};


/**
 * This template implements the truncated (or split) storage of a floating point
 * type.
 *
 * The class splits the type FloatType into NumComponents components. `i`-th
 * component of the class is represented by the instantiation with ComponentId
 * equal to `i`. 0-th component is the one with most significant bits, i.e. the
 * one that includes the sign and the exponent of the number.
 *
 * @tparam FloatType  a floating point type this truncates / splits
 * @tparam NumComponents  number of equally-sized components FloatType is split
 *                        into
 * @tparam ComponentId  index of the component of FloatType an object of this
 *                      template instntiation represents
 */
template <typename FloatType, size_type NumComponents,
          size_type ComponentId = 0>
class truncated {
public:
    using float_type = FloatType;

    /**
     * Unsigned type representing the bits of FloatType.
     */
    using full_bits_type = typename detail::float_traits<float_type>::bits_type;

    static constexpr auto num_components = NumComponents;
    static constexpr auto component_id = ComponentId;

    /**
     * Size of the component in bits.
     */
    static constexpr auto component_size =
        sizeof(float_type) * byte_size / num_components;
    /**
     * Starting bit position of the component in FloatType.
     */
    static constexpr auto component_position =
        (num_components - component_id - 1) * component_size;
    /**
     * Bitmask of the component in FloatType.
     */
    static constexpr auto component_mask =
        detail::create_ones<full_bits_type>(component_size)
        << component_position;

    /**
     * Unsigned type representing the bits of the component.
     */
    using bits_type = detail::uint_of<component_size>;

    static_assert((sizeof(float_type) * byte_size) % component_size == 0,
                  "Size of float is not a multiple of component size");
    static_assert(component_id < num_components,
                  "This type doesn't have that many components");

    GKO_ATTRIBUTES truncated() noexcept = default;

    GKO_ATTRIBUTES explicit truncated(const float_type &val) noexcept
    {
        const auto &bits = reinterpret_cast<const full_bits_type &>(val);
        data_ = static_cast<bits_type>((bits & component_mask) >>
                                       component_position);
    }

    GKO_ATTRIBUTES operator float_type() const noexcept
    {
        const auto bits = static_cast<full_bits_type>(data_)
                          << component_position;
        return reinterpret_cast<const float_type &>(bits);
    }

private:
    bits_type data_;
};


}  // namespace gko


namespace std {


template <>
class complex<gko::half> {
public:
    using value_type = gko::half;

    complex(const value_type &real = 0.f, const value_type &imag = 0.f)
        : real_(real), imag_(imag)
    {}

    template <typename U>
    explicit complex(const complex<U> &other)
        : complex(static_cast<value_type>(other.real()),
                  static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }


    operator std::complex<gko::float32>() const noexcept
    {
        return std::complex<gko::float32>(static_cast<gko::float32>(real_),
                                          static_cast<gko::float32>(imag_));
    }

private:
    value_type real_;
    value_type imag_;
};


template <typename T, gko::size_type NumComponents>
class complex<gko::truncated<T, NumComponents>> {
public:
    using value_type = gko::truncated<T, NumComponents>;

    complex(const value_type &real = 0.f, const value_type &imag = 0.f)
        : real_(real), imag_(imag)
    {}

    template <typename U>
    explicit complex(const complex<U> &other)
        : complex(static_cast<value_type>(other.real()),
                  static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }


    operator std::complex<T>() const noexcept
    {
        return std::complex<T>(static_cast<T>(real_), static_cast<T>(imag_));
    }

private:
    value_type real_;
    value_type imag_;
};


}  // namespace std


#endif  // GKO_CORE_BASE_EXTENDED_FLOAT_HPP_
