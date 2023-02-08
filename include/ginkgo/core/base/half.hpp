/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_PUBLIC_CORE_BASE_HALF_HPP_
#define GKO_PUBLIC_CORE_BASE_HALF_HPP_


#include <complex>
#include <type_traits>


#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/base/types.hpp>

#ifdef SYCL_LANGUAGE_VERSION
#include <CL/sycl.hpp>
#endif

#ifdef __CUDA_ARCH__


#include <cuda_fp16.h>


#elif defined(__HIP_DEVICE_COMPILE__)


#include <hip/hip_fp16.h>


#else


class __half;


#endif  // __CUDA_ARCH__


namespace gko {


template <typename, size_type, size_type>
class truncated;


namespace detail {


template <std::size_t, typename = void>
struct uint_of_impl {};

template <std::size_t Bits>
struct uint_of_impl<Bits, std::enable_if_t<(Bits <= 16)>> {
    using type = uint16;
};

template <std::size_t Bits>
struct uint_of_impl<Bits, std::enable_if_t<(16 < Bits && Bits <= 32)>> {
    using type = uint32;
};

template <std::size_t Bits>
struct uint_of_impl<Bits, std::enable_if_t<(32 < Bits)>> {
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

// #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
template <>
struct basic_float_traits<__half> {
    using type = __half;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 10;
    static constexpr int exponent_bits = 5;
    static constexpr bool rounds_to_nearest = true;
};
// #endif

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

#ifdef SYCL_LANGUAGE_VERSION
using half = sycl::half;
#else
/**
 * A class providing basic support for half precision floating point types.
 *
 * For now the only features are reduced storage compared to single precision
 * and conversions from and to single precision floating point type.
 */
class half {
public:
    GKO_ATTRIBUTES half() noexcept = default;

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    GKO_ATTRIBUTES half(const T val)
    {
        this->float2half(static_cast<float>(val));
    }

    GKO_ATTRIBUTES half(const half& val) = default;

    template <typename V>
    GKO_ATTRIBUTES half& operator=(const V val)
    {
        this->float2half(static_cast<float>(val));
        return *this;
    }

    GKO_ATTRIBUTES operator float() const noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return __half2float(reinterpret_cast<const __half&>(data_));
#else   // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const auto bits = half2float(data_);
        return reinterpret_cast<const float32&>(bits);
#endif  // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }

    // can not use half operator _op(const half) for half + half
    // operation will cast it to float and then do float operation such that it
    // becomes float in the end.
#define HALF_OPERATOR(_op, _opeq)                                           \
    GKO_ATTRIBUTES friend half operator _op(const half lhf, const half rhf) \
    {                                                                       \
        return static_cast<half>(static_cast<float>(lhf)                    \
                                     _op static_cast<float>(rhf));          \
    }                                                                       \
    GKO_ATTRIBUTES half& operator _opeq(const half& hf)                     \
    {                                                                       \
        auto result = *this _op hf;                                         \
        this->float2half(result);                                           \
        return *this;                                                       \
    }
    HALF_OPERATOR(+, +=)
    HALF_OPERATOR(-, -=)
    HALF_OPERATOR(*, *=)
    HALF_OPERATOR(/, /=)

    // Do operation with different type
    // If it is floating point, using floating point as type.
    // If it is integer, using half as type
#define HALF_FRIEND_OPERATOR(_op, _opeq)                                   \
    template <typename T>                                                  \
    GKO_ATTRIBUTES friend std::enable_if_t<                                \
        !std::is_same<T, half>::value && std::is_scalar<T>::value,         \
        typename std::conditional<std::is_floating_point<T>::value, T,     \
                                  half>::type>                             \
    operator _op(const half hf, const T val)                               \
    {                                                                      \
        using type =                                                       \
            typename std::conditional<std::is_floating_point<T>::value, T, \
                                      half>::type;                         \
        auto result = static_cast<type>(hf);                               \
        result _opeq static_cast<type>(val);                               \
        return result;                                                     \
    }                                                                      \
    template <typename T>                                                  \
    GKO_ATTRIBUTES friend std::enable_if_t<                                \
        !std::is_same<T, half>::value && std::is_scalar<T>::value,         \
        typename std::conditional<std::is_floating_point<T>::value, T,     \
                                  half>::type>                             \
    operator _op(const T val, const half hf)                               \
    {                                                                      \
        using type =                                                       \
            typename std::conditional<std::is_floating_point<T>::value, T, \
                                      half>::type;                         \
        auto result = static_cast<type>(val);                              \
        result _opeq static_cast<type>(hf);                                \
        return result;                                                     \
    }

    HALF_FRIEND_OPERATOR(+, +=)
    HALF_FRIEND_OPERATOR(-, -=)
    HALF_FRIEND_OPERATOR(*, *=)
    HALF_FRIEND_OPERATOR(/, /=)

    // the negative
    GKO_ATTRIBUTES half operator-() const
    {
        auto val = 0.0f - *this;
        return half(val);
    }

private:
    using f16_traits = detail::float_traits<float16>;
    using f32_traits = detail::float_traits<float32>;

    // TODO: do we really need this one?
    // Without it, everything can be constexpr, which might make stuff easier.
    GKO_ATTRIBUTES void float2half(float val) noexcept
    {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        const auto tmp = __float2half_rn(val);
        data_ = reinterpret_cast<const uint16&>(tmp);
#else   // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        data_ = float2half(reinterpret_cast<const uint32&>(val));
#endif  // defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    }

    static GKO_ATTRIBUTES uint16 float2half(uint32 data_) noexcept
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

    static GKO_ATTRIBUTES uint32 half2float(uint16 data_) noexcept
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
#endif


}  // namespace gko


namespace std {


template <>
class complex<gko::half> {
public:
    using value_type = gko::half;

    complex(const value_type& real = value_type(0.f),
            const value_type& imag = value_type(0.f))
        : real_(real), imag_(imag)
    {}
    template <typename T, typename U,
              typename = std::enable_if_t<std::is_scalar<T>::value &&
                                          std::is_scalar<U>::value>>
    explicit complex(const T& real, const U& imag)
        : complex(static_cast<value_type>(real), static_cast<value_type>(imag))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const T& real) : complex(static_cast<value_type>(real))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    explicit complex(const complex<T>& other)
        : complex(static_cast<value_type>(other.real()),
                  static_cast<value_type>(other.imag()))
    {}

    // explicit complex(const complex<value_type>& other) = default;

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }


    operator std::complex<float>() const noexcept
    {
        return std::complex<float>(static_cast<float>(real_),
                                   static_cast<float>(imag_));
    }

    // operator std::complex<double>() const noexcept
    // {
    //     return std::complex<double>(static_cast<double>(real_),
    //                                 static_cast<double>(imag_));
    // }

    template <typename V>
    complex& operator=(const V& val)
    {
        real_ = val;
        imag_ = value_type();
        return *this;
    }

    template <typename V>
    complex& operator=(const std::complex<V>& val)
    {
        real_ = val.real();
        imag_ = val.imag();
        return *this;
    }

    complex& operator+=(const value_type& real)
    {
        real_ += real;
        return *this;
    }
    complex& operator-=(const value_type& real)
    {
        real_ -= real;
        return *this;
    }
    complex& operator*=(const value_type& real)
    {
        real_ *= real;
        imag_ *= real;
        return *this;
    }
    complex& operator/=(const value_type& real)
    {
        real_ /= real;
        imag_ /= real;
        return *this;
    }

    template <typename T>
    complex& operator+=(const complex<T>& val)
    {
        real_ += val.real();
        imag_ += val.imag();
        return *this;
    }
    template <typename T>
    complex& operator-=(const complex<T>& val)
    {
        real_ -= val.real();
        imag_ -= val.imag();
        return *this;
    }
    template <typename T>
    complex& operator*=(const complex<T>& val)
    {
        auto tmp = real_;
        real_ = real_ * val.real() - imag_ * val.imag();
        imag_ = tmp * val.imag() + imag_ * val.real();
        return *this;
    }
    template <typename T>
    complex& operator/=(const complex<T>& val)
    {
        auto real = val.real();
        auto imag = val.imag();
        (*this) *= complex<T>{val.real(), -val.imag()};
        (*this) /= (real * real + imag * imag);
        return *this;
    }

// It's for MacOS.
// TODO: check whether mac compiler always use complex version even when real
// half
#define COMPLEX_HALF_OPERATOR(_op, _opeq)                           \
    GKO_ATTRIBUTES friend complex<gko::half> operator _op(          \
        const complex<gko::half> lhf, const complex<gko::half> rhf) \
    {                                                               \
        auto a = lhf;                                               \
        a _opeq rhf;                                                \
        return a;                                                   \
    }

    COMPLEX_HALF_OPERATOR(+, +=)
    COMPLEX_HALF_OPERATOR(-, -=)
    COMPLEX_HALF_OPERATOR(*, *=)
    COMPLEX_HALF_OPERATOR(/, /=)

private:
    value_type real_;
    value_type imag_;
};

#ifndef SYCL_LANGUAGE_VERSION
template <>
struct numeric_limits<gko::half> {
    static constexpr bool is_specialized{true};
    static constexpr bool is_signed{true};
    static constexpr bool is_integer{false};
    static constexpr bool is_exact{false};
    static constexpr bool is_bounded{true};
    static constexpr bool is_modulo{false};
    static constexpr int digits{
        gko::detail::float_traits<gko::half>::significand_bits + 1};
    // 3/10 is approx. log_10(2)
    static constexpr int digits10{digits * 3 / 10};

    // Note: gko::half can't return gko::half here because it does not have
    //       a constexpr constructor.
    static constexpr float epsilon()
    {
        return gko::detail::float_traits<gko::half>::eps;
    }

    static constexpr float infinity()
    {
        return numeric_limits<float>::infinity();
    }

    static constexpr float min() { return numeric_limits<float>::min(); }

    static constexpr float max() { return numeric_limits<float>::max(); }

    static constexpr float quiet_NaN()
    {
        return numeric_limits<float>::quiet_NaN();
    }
};

#endif

// complex using a template on operator= for any kind of complex<T>, so we can
// do full specialization for half
template <>
inline complex<double>& complex<double>::operator=(
    const std::complex<gko::half>& a)
{
    complex<double> t(a.real(), a.imag());
    operator=(t);
    return *this;
}


// For MSVC
template <>
inline complex<float>& complex<float>::operator=(
    const std::complex<gko::half>& a)
{
    complex<float> t(a.real(), a.imag());
    operator=(t);
    return *this;
}


}  // namespace std


#endif  // GKO_PUBLIC_CORE_BASE_HALF_HPP_
