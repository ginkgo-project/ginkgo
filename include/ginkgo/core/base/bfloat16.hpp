// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_BFLOAT16_HPP_
#define GKO_PUBLIC_CORE_BASE_BFLOAT16_HPP_


#include <climits>
#include <complex>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <ginkgo/core/base/half.hpp>


class __nv_bfloat16;
class hip_bfloat16;
class __hip_bfloat16;


namespace gko {


class bfloat16;


namespace detail {
template <>
struct basic_float_traits<bfloat16> {
    using type = bfloat16;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 7;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<__nv_bfloat16> {
    using type = __nv_bfloat16;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 7;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<hip_bfloat16> {
    using type = hip_bfloat16;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 7;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<__hip_bfloat16> {
    using type = __hip_bfloat16;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 7;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};


}  // namespace detail


/**
 * A class providing basic support for bfloat16 precision floating point types.
 *
 * For now the only features are reduced storage compared to single precision
 * and conversions from and to single precision floating point type.
 */
class alignas(std::uint16_t) bfloat16 {
public:
    // create bfloat16 value from the bits directly.
    static constexpr bfloat16 create_from_bits(
        const std::uint16_t& bits) noexcept
    {
        bfloat16 result;
        result.data_ = bits;
        return result;
    }

    // TODO: NVHPC (host side) may not use zero initialization for the data
    // member by default constructor in some cases. Not sure whether it is
    // caused by something else in jacobi or isai.
    constexpr bfloat16() noexcept : data_(0){};

    template <typename T,
              typename = std::enable_if_t<std::is_scalar<T>::value ||
                                          std::is_same_v<T, half>>>
    bfloat16(const T& val) : data_(0)
    {
        this->float2bfloat16(static_cast<float>(val));
    }

    template <typename V>
    bfloat16& operator=(const V& val)
    {
        this->float2bfloat16(static_cast<float>(val));
        return *this;
    }

    operator float() const noexcept
    {
        const auto bits = bfloat162float(data_);
        float ans(0);
        std::memcpy(&ans, &bits, sizeof(float));
        return ans;
    }

    // can not use bfloat16 operator _op(const bfloat16) for bfloat16 + bfloat16
    // operation will cast it to float and then do float operation such that it
    // becomes float in the end.
#define BFLOAT16_OPERATOR(_op, _opeq)                                      \
    friend bfloat16 operator _op(const bfloat16& lhf, const bfloat16& rhf) \
    {                                                                      \
        return static_cast<bfloat16>(static_cast<float>(lhf)               \
                                         _op static_cast<float>(rhf));     \
    }                                                                      \
    bfloat16& operator _opeq(const bfloat16& hf)                           \
    {                                                                      \
        auto result = *this _op hf;                                        \
        data_ = result.data_;                                              \
        return *this;                                                      \
    }

    BFLOAT16_OPERATOR(+, +=)
    BFLOAT16_OPERATOR(-, -=)
    BFLOAT16_OPERATOR(*, *=)
    BFLOAT16_OPERATOR(/, /=)

#undef BFLOAT16_OPERATOR

    // Do operation with different type
    // If it is floating point, using floating point as type.
    // If it is bfloat16, using float as type.
    // If it is integer, using bfloat16 as type.
#define BFLOAT16_FRIEND_OPERATOR(_op, _opeq)                                   \
    template <typename T>                                                      \
    friend std::enable_if_t<                                                   \
        !std::is_same<T, bfloat16>::value &&                                   \
            (std::is_scalar<T>::value || std::is_same_v<T, half>),             \
        std::conditional_t<                                                    \
            std::is_floating_point<T>::value, T,                               \
            std::conditional_t<std::is_same_v<T, half>, float, bfloat16>>>     \
    operator _op(const bfloat16& hf, const T& val)                             \
    {                                                                          \
        using type =                                                           \
            std::conditional_t<std::is_floating_point<T>::value, T, bfloat16>; \
        auto result = static_cast<type>(hf);                                   \
        result _opeq static_cast<type>(val);                                   \
        return result;                                                         \
    }                                                                          \
    template <typename T>                                                      \
    friend std::enable_if_t<                                                   \
        !std::is_same<T, bfloat16>::value &&                                   \
            (std::is_scalar<T>::value || std::is_same_v<T, half>),             \
        std::conditional_t<                                                    \
            std::is_floating_point<T>::value, T,                               \
            std::conditional_t<std::is_same_v<T, half>, float, bfloat16>>>     \
    operator _op(const T& val, const bfloat16& hf)                             \
    {                                                                          \
        using type =                                                           \
            std::conditional_t<std::is_floating_point<T>::value, T, bfloat16>; \
        auto result = static_cast<type>(val);                                  \
        result _opeq static_cast<type>(hf);                                    \
        return result;                                                         \
    }

    BFLOAT16_FRIEND_OPERATOR(+, +=)
    BFLOAT16_FRIEND_OPERATOR(-, -=)
    BFLOAT16_FRIEND_OPERATOR(*, *=)
    BFLOAT16_FRIEND_OPERATOR(/, /=)

#undef BFLOAT16_FRIEND_OPERATOR

    // the negative
    bfloat16 operator-() const
    {
        auto val = 0.0f - *this;
        return static_cast<bfloat16>(val);
    }

private:
    using f16_traits = detail::float_traits<bfloat16>;
    using f32_traits = detail::float_traits<float>;

    void float2bfloat16(const float& val) noexcept
    {
        std::uint32_t bit_val(0);
        std::memcpy(&bit_val, &val, sizeof(float));
        data_ = float2bfloat16(bit_val);
    }

    static constexpr std::uint16_t float2bfloat16(std::uint32_t data_) noexcept
    {
        using conv = detail::precision_converter<float, bfloat16>;
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
                // Rounding to even
                const auto result = conv::shift_sign(data_) | exp |
                                    conv::shift_significand(data_);
                const auto tail =
                    data_ & static_cast<f32_traits::bits_type>(
                                (1 << conv::significand_offset) - 1);

                constexpr auto bfloat16 = static_cast<f32_traits::bits_type>(
                    1 << (conv::significand_offset - 1));
                return result + (tail > bfloat16 ||
                                 ((tail == bfloat16) && (result & 1)));
            }
        }
    }

    static constexpr std::uint32_t bfloat162float(std::uint16_t data_) noexcept
    {
        using conv = detail::precision_converter<bfloat16, float>;
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

    std::uint16_t data_;
};


}  // namespace gko


namespace std {


template <>
class complex<gko::bfloat16> {
public:
    using value_type = gko::bfloat16;

    complex(const value_type& real = value_type(0.f),
            const value_type& imag = value_type(0.f))
        : real_(real), imag_(imag)
    {}

    template <
        typename T, typename U,
        typename = std::enable_if_t<
            (std::is_scalar<T>::value || std::is_same_v<T, gko::half>)&&(
                std::is_scalar<U>::value || std::is_same_v<U, gko::half>)>>
    explicit complex(const T& real, const U& imag)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(imag))
    {}

    template <typename T,
              typename = std::enable_if_t<std::is_scalar<T>::value ||
                                          std::is_same_v<T, gko::half>>>
    complex(const T& real)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(0.f))
    {}

    // When using complex(real, imag), MSVC with CUDA try to recognize the
    // complex is a member not constructor.
    template <typename T,
              typename = std::enable_if_t<std::is_scalar<T>::value ||
                                          std::is_same_v<T, gko::half>>>
    explicit complex(const complex<T>& other)
        : real_(static_cast<value_type>(other.real())),
          imag_(static_cast<value_type>(other.imag()))
    {}

    value_type real() const noexcept { return real_; }

    value_type imag() const noexcept { return imag_; }

    operator std::complex<float>() const noexcept
    {
        return std::complex<float>(static_cast<float>(real_),
                                   static_cast<float>(imag_));
    }

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
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f *= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

    template <typename T>
    complex& operator/=(const complex<T>& val)
    {
        auto val_f = static_cast<std::complex<float>>(val);
        auto result_f = static_cast<std::complex<float>>(*this);
        result_f /= val_f;
        real_ = result_f.real();
        imag_ = result_f.imag();
        return *this;
    }

#define COMPLEX_BFLOAT16_OPERATOR(_op, _opeq)                           \
    friend complex operator _op(const complex& lhf, const complex& rhf) \
    {                                                                   \
        auto a = lhf;                                                   \
        a _opeq rhf;                                                    \
        return a;                                                       \
    }

    COMPLEX_BFLOAT16_OPERATOR(+, +=)
    COMPLEX_BFLOAT16_OPERATOR(-, -=)
    COMPLEX_BFLOAT16_OPERATOR(*, *=)
    COMPLEX_BFLOAT16_OPERATOR(/, /=)

#undef COMPLEX_BFLOAT16_OPERATOR

private:
    value_type real_;
    value_type imag_;
};


template <>
struct numeric_limits<gko::bfloat16> {
    static constexpr bool is_specialized{true};
    static constexpr bool is_signed{true};
    static constexpr bool is_integer{false};
    static constexpr bool is_exact{false};
    static constexpr bool is_bounded{true};
    static constexpr bool is_modulo{false};
    static constexpr int digits{
        gko::detail::float_traits<gko::bfloat16>::significand_bits + 1};
    // 3/10 is approx. log_10(2)
    static constexpr int digits10{digits * 3 / 10};

    static constexpr gko::bfloat16 epsilon()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'01111000'0000000u);
        return gko::bfloat16::create_from_bits(bits);
    }

    static constexpr gko::bfloat16 infinity()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'11111111'0000000u);
        return gko::bfloat16::create_from_bits(bits);
    }

    static constexpr gko::bfloat16 min()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'00000001'0000000u);
        return gko::bfloat16::create_from_bits(bits);
    }

    static constexpr gko::bfloat16 max()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'11111110'1111111u);
        return gko::bfloat16::create_from_bits(bits);
    }

    static constexpr gko::bfloat16 lowest()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b1'11111110'1111111u);
        return gko::bfloat16::create_from_bits(bits);
    };

    static constexpr gko::bfloat16 quiet_NaN()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'11111111'1111111u);
        return gko::bfloat16::create_from_bits(bits);
    }
};


// complex using a template on operator= for any kind of complex<T>, so we can
// do full specialization for bfloat16
template <>
inline complex<double>& complex<double>::operator=(
    const std::complex<gko::bfloat16>& a)
{
    complex<double> t(a.real(), a.imag());
    operator=(t);
    return *this;
}


// For MSVC
template <>
inline complex<float>& complex<float>::operator=(
    const std::complex<gko::bfloat16>& a)
{
    complex<float> t(a.real(), a.imag());
    operator=(t);
    return *this;
}


}  // namespace std


#endif  // GKO_PUBLIC_CORE_BASE_bfloat16_HPP_
