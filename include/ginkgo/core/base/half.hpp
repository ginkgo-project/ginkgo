// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_HALF_HPP_
#define GKO_PUBLIC_CORE_BASE_HALF_HPP_


#include <climits>
#include <complex>
#include <cstdint>
#include <cstring>
#include <type_traits>


class __half;


namespace gko {


template <typename, std::size_t, std::size_t>
class truncated;


class half;


namespace detail {


constexpr std::size_t byte_size = CHAR_BIT;

template <std::size_t, typename = void>
struct uint_of_impl {};

template <std::size_t Bits>
struct uint_of_impl<Bits, std::enable_if_t<(Bits <= 16)>> {
    using type = std::uint16_t;
};

template <std::size_t Bits>
struct uint_of_impl<Bits, std::enable_if_t<(16 < Bits && Bits <= 32)>> {
    using type = std::uint32_t;
};

template <std::size_t Bits>
struct uint_of_impl<Bits, std::enable_if_t<(32 < Bits) && (Bits <= 64)>> {
    using type = std::uint64_t;
};

template <std::size_t Bits>
using uint_of = typename uint_of_impl<Bits>::type;


template <typename T>
struct basic_float_traits {};

template <>
struct basic_float_traits<half> {
    using type = half;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 10;
    static constexpr int exponent_bits = 5;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<__half> {
    using type = __half;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 10;
    static constexpr int exponent_bits = 5;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<float> {
    using type = float;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 23;
    static constexpr int exponent_bits = 8;
    static constexpr bool rounds_to_nearest = true;
};

template <>
struct basic_float_traits<double> {
    using type = double;
    static constexpr int sign_bits = 1;
    static constexpr int significand_bits = 52;
    static constexpr int exponent_bits = 11;
    static constexpr bool rounds_to_nearest = true;
};

template <typename FloatType, std::size_t NumComponents,
          std::size_t ComponentId>
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
class alignas(std::uint16_t) half {
public:
    // create half value from the bits directly.
    static constexpr half create_from_bits(const std::uint16_t& bits) noexcept
    {
        half result;
        result.data_ = bits;
        return result;
    }

    // TODO: NVHPC (host side) may not use zero initialization for the data
    // member by default constructor in some cases. Not sure whether it is
    // caused by something else in jacobi or isai.
    constexpr half() noexcept : data_(0){};

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    half(const T& val) : data_(0)
    {
        this->float2half(static_cast<float>(val));
    }

    template <typename V>
    half& operator=(const V& val)
    {
        this->float2half(static_cast<float>(val));
        return *this;
    }

    operator float() const noexcept
    {
        const auto bits = half2float(data_);
        float ans(0);
        std::memcpy(&ans, &bits, sizeof(float));
        return ans;
    }

    // can not use half operator _op(const half) for half + half
    // operation will cast it to float and then do float operation such that it
    // becomes float in the end.
#define HALF_OPERATOR(_op, _opeq)                                  \
    friend half operator _op(const half& lhf, const half& rhf)     \
    {                                                              \
        return static_cast<half>(static_cast<float>(lhf)           \
                                     _op static_cast<float>(rhf)); \
    }                                                              \
    half& operator _opeq(const half& hf)                           \
    {                                                              \
        auto result = *this _op hf;                                \
        data_ = result.data_;                                      \
        return *this;                                              \
    }

    HALF_OPERATOR(+, +=)
    HALF_OPERATOR(-, -=)
    HALF_OPERATOR(*, *=)
    HALF_OPERATOR(/, /=)

#undef HALF_OPERATOR

    // Do operation with different type
    // If it is floating point, using floating point as type.
    // If it is integer, using half as type
#define HALF_FRIEND_OPERATOR(_op, _opeq)                                   \
    template <typename T>                                                  \
    friend std::enable_if_t<                                               \
        !std::is_same<T, half>::value && std::is_scalar<T>::value,         \
        std::conditional_t<std::is_floating_point<T>::value, T, half>>     \
    operator _op(const half& hf, const T& val)                             \
    {                                                                      \
        using type =                                                       \
            std::conditional_t<std::is_floating_point<T>::value, T, half>; \
        auto result = static_cast<type>(hf);                               \
        result _opeq static_cast<type>(val);                               \
        return result;                                                     \
    }                                                                      \
    template <typename T>                                                  \
    friend std::enable_if_t<                                               \
        !std::is_same<T, half>::value && std::is_scalar<T>::value,         \
        std::conditional_t<std::is_floating_point<T>::value, T, half>>     \
    operator _op(const T& val, const half& hf)                             \
    {                                                                      \
        using type =                                                       \
            std::conditional_t<std::is_floating_point<T>::value, T, half>; \
        auto result = static_cast<type>(val);                              \
        result _opeq static_cast<type>(hf);                                \
        return result;                                                     \
    }

    HALF_FRIEND_OPERATOR(+, +=)
    HALF_FRIEND_OPERATOR(-, -=)
    HALF_FRIEND_OPERATOR(*, *=)
    HALF_FRIEND_OPERATOR(/, /=)

#undef HALF_FRIEND_OPERATOR

    // the negative
    half operator-() const
    {
        auto val = 0.0f - *this;
        return static_cast<half>(val);
    }

private:
    using f16_traits = detail::float_traits<half>;
    using f32_traits = detail::float_traits<float>;

    void float2half(const float& val) noexcept
    {
        std::uint32_t bit_val(0);
        std::memcpy(&bit_val, &val, sizeof(float));
        data_ = float2half(bit_val);
    }

    static constexpr std::uint16_t float2half(std::uint32_t data_) noexcept
    {
        using conv = detail::precision_converter<float, half>;
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

                constexpr auto half = static_cast<f32_traits::bits_type>(
                    1 << (conv::significand_offset - 1));
                return result +
                       (tail > half || ((tail == half) && (result & 1)));
            }
        }
    }

    static constexpr std::uint32_t half2float(std::uint16_t data_) noexcept
    {
        using conv = detail::precision_converter<half, float>;
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
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(imag))
    {}

    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
    complex(const T& real)
        : real_(static_cast<value_type>(real)),
          imag_(static_cast<value_type>(0.f))
    {}

    // When using complex(real, imag), MSVC with CUDA try to recognize the
    // complex is a member not constructor.
    template <typename T, typename = std::enable_if_t<std::is_scalar<T>::value>>
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

#define COMPLEX_HALF_OPERATOR(_op, _opeq)                               \
    friend complex operator _op(const complex& lhf, const complex& rhf) \
    {                                                                   \
        auto a = lhf;                                                   \
        a _opeq rhf;                                                    \
        return a;                                                       \
    }

    COMPLEX_HALF_OPERATOR(+, +=)
    COMPLEX_HALF_OPERATOR(-, -=)
    COMPLEX_HALF_OPERATOR(*, *=)
    COMPLEX_HALF_OPERATOR(/, /=)

#undef COMPLEX_HALF_OPERATOR

private:
    value_type real_;
    value_type imag_;
};


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

    static constexpr gko::half epsilon()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'00101'0000000000u);
        return gko::half::create_from_bits(bits);
    }

    static constexpr gko::half infinity()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'11111'0000000000u);
        return gko::half::create_from_bits(bits);
    }

    static constexpr gko::half min()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'00001'0000000000u);
        return gko::half::create_from_bits(bits);
    }

    static constexpr gko::half max()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'11110'1111111111u);
        return gko::half::create_from_bits(bits);
    }

    static constexpr gko::half lowest()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b1'11110'1111111111u);
        return gko::half::create_from_bits(bits);
    };

    static constexpr gko::half quiet_NaN()
    {
        constexpr auto bits = static_cast<std::uint16_t>(0b0'11111'1111111111u);
        return gko::half::create_from_bits(bits);
    }
};


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
