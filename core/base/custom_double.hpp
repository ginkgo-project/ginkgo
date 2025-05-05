// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_DOUBLE_HPP_
#define GKO_CORE_BASE_DOUBLE_HPP_


#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/types.hpp>

#ifdef GKO_COMPILING_CUDA
#include <cuda_fp16.h>
#endif


namespace gko {

/**
 * A class providing basic support for double precision floating point types.
 *
 * For now the only features are reduced storage compared to single precision
 * and conversions from and to single precision floating point type.
 */
class alignas(std::uint64_t) custom_double {
public:
    // create half value from the bits directly.
    static constexpr custom_double create_from_bits(
        const std::uint64_t& bits) noexcept
    {
        custom_double result;
        result.data_ = bits;
        return result;
    }

    // TODO: NVHPC (host side) may not use zero initialization for the data
    // member by default constructor in some cases. Not sure whether it is
    // caused by something else in jacobi or isai.
    constexpr custom_double() noexcept : data_(0){};

    // avoid the conversion available public to double such that we do not
    // accendiently convert to double no double input to be sure
    template <typename T,
              typename = std::enable_if_t<(!std::is_same<T, double>::value &&
                                           std::is_scalar<T>::value) ||
                                          std::is_same<T, half>::value>>
    GKO_ATTRIBUTES GKO_INLINE custom_double(const T& val) : data_(0)
    {
        // TODO: do not use static_cast double
        data_ = native_to_custom(static_cast<double>(val));
    }

    // TODO: this is firstly required by the thrust complex operation
    // T(1.0)/custom_double.
    explicit GKO_ATTRIBUTES GKO_INLINE custom_double(const double& val)
    {
        data_ = native_to_custom(val);
    }

    GKO_ATTRIBUTES GKO_INLINE operator float() const noexcept
    {
        return static_cast<float>(custom_to_native(data_));
    }

    GKO_ATTRIBUTES GKO_INLINE operator gko::int64() const noexcept
    {
        return static_cast<gko::int64>(custom_to_native(data_));
    }

    GKO_ATTRIBUTES GKO_INLINE operator gko::int32() const noexcept
    {
        return static_cast<gko::int32>(custom_to_native(data_));
    }

    GKO_ATTRIBUTES GKO_INLINE operator gko::int16() const noexcept
    {
        return static_cast<gko::int16>(custom_to_native(data_));
    }

#ifdef GKO_COMPILING_CUDA
    GKO_ATTRIBUTES GKO_INLINE operator __half() const noexcept
    {
        return static_cast<__half>(custom_to_native(data_));
    }
#endif

    // can not use custom_double operator _op(const custom_double) for
    // custom_double + custom_double operation will cast it to float and then do
    // float operation such that it becomes float in the end.
    // TODO replace by proper simulation
#define CUSTOM_DOUBLE_OPERATOR(_op, _opeq)                       \
    friend GKO_ATTRIBUTES GKO_INLINE custom_double operator _op( \
        const custom_double& lhf, const custom_double& rhf)      \
    {                                                            \
        return custom_double::native_to_custom(                  \
            custom_double::custom_to_native(lhf.data_)           \
                _op custom_double::custom_to_native(rhf.data_)); \
    }                                                            \
    GKO_ATTRIBUTES GKO_INLINE custom_double& operator _opeq(     \
        const custom_double& hf)                                 \
    {                                                            \
        auto result = *this _op hf;                              \
        data_ = result.data_;                                    \
        return *this;                                            \
    }

    CUSTOM_DOUBLE_OPERATOR(+, +=)
    CUSTOM_DOUBLE_OPERATOR(-, -=)
    CUSTOM_DOUBLE_OPERATOR(*, *=)
    CUSTOM_DOUBLE_OPERATOR(/, /=)

#undef CUSTOM_DOUBLE_OPERATOR

    // the negative
    GKO_ATTRIBUTES GKO_INLINE custom_double operator-() const
    {
        // TODO: simulate by int64
        auto val = 0.0 - custom_to_native(data_);
        return native_to_custom(val);
    }

#define CUSTOM_DOUBLE_COMPARTOR(_op)                        \
    friend GKO_ATTRIBUTES GKO_INLINE bool operator _op(     \
        const custom_double& lhf, const custom_double& rhf) \
    {                                                       \
        return custom_double::custom_to_native(lhf)         \
            _op custom_double::custom_to_native(rhf);       \
    }

    CUSTOM_DOUBLE_COMPARTOR(==)
    CUSTOM_DOUBLE_COMPARTOR(!=)
    CUSTOM_DOUBLE_COMPARTOR(<)
    CUSTOM_DOUBLE_COMPARTOR(<=)
    CUSTOM_DOUBLE_COMPARTOR(>)
    CUSTOM_DOUBLE_COMPARTOR(>=)

#undef CUSTOM_DOUBLE_COMPARTOR

    static GKO_ATTRIBUTES GKO_INLINE std::uint64_t native_to_custom(
        const double& val) noexcept
    {
        std::uint64_t data;
        memcpy(&data, &val, sizeof(double));
        return data;
    }

    static GKO_ATTRIBUTES GKO_INLINE custom_double
    to_custom(const double& val) noexcept
    {
        custom_double data;
        memcpy(&data.data_, &val, sizeof(double));
        return data;
    }

    static GKO_ATTRIBUTES GKO_INLINE double custom_to_native(
        const std::uint64_t& data) noexcept
    {
        double result;
        memcpy(&result, &data, sizeof(double));
        return result;
    }

    static GKO_ATTRIBUTES GKO_INLINE double custom_to_native(
        const custom_double& val) noexcept
    {
        double result;
        memcpy(&result, &val.data_, sizeof(double));
        return result;
    }

private:
    // leave it to private such that we do call outside?
    operator double() const noexcept { return custom_to_native(data_); }

    std::uint64_t data_;
};


}  // namespace gko


namespace std {


// We do not need the complex varient because we only use custom_double on
// device side and we will use the thrust::complex.


template <>
struct numeric_limits<gko::custom_double> {
    static constexpr bool is_specialized{true};
    static constexpr bool is_signed{true};
    static constexpr bool is_integer{false};
    static constexpr bool is_exact{false};
    static constexpr bool is_bounded{true};
    static constexpr bool is_modulo{false};
    static constexpr int digits{
        gko::detail::basic_float_traits<gko::custom_double>::significand_bits +
        1};
    // 3/10 is approx. log_10(2)
    static constexpr int digits10{digits * 3 / 10};

    static constexpr gko::custom_double epsilon()
    {
        constexpr auto bits = static_cast<std::uint64_t>(
            0b0'00000000101'0000000000000000000000000000000000000000000000000000ull);
        return gko::custom_double::create_from_bits(bits);
    }

    static constexpr gko::custom_double infinity()
    {
        constexpr auto bits = static_cast<std::uint64_t>(
            0b0'11111111111'0000000000000000000000000000000000000000000000000000ull);
        return gko::custom_double::create_from_bits(bits);
    }

    static constexpr gko::custom_double min()
    {
        constexpr auto bits = static_cast<std::uint64_t>(
            0b0'00000000001'0000000000000000000000000000000000000000000000000000ull);
        return gko::custom_double::create_from_bits(bits);
    }

    static constexpr gko::custom_double max()
    {
        constexpr auto bits = static_cast<std::uint16_t>(
            0b0'11111111110'1111111111111111111111111111111111111111111111111111ull);
        return gko::custom_double::create_from_bits(bits);
    }

    static constexpr gko::custom_double lowest()
    {
        constexpr auto bits = static_cast<std::uint16_t>(
            0b1'11111111110'1111111111111111111111111111111111111111111111111111ull);
        return gko::custom_double::create_from_bits(bits);
    };

    static constexpr gko::custom_double quiet_NaN()
    {
        constexpr auto bits = static_cast<std::uint16_t>(
            0b0'11111111111'1111111111111111111111111111111111111111111111111111ull);
        return gko::custom_double::create_from_bits(bits);
    }
};


}  // namespace std


namespace gko::detail {
template <>
struct float_traits<custom_double> {
    using type = typename basic_float_traits<custom_double>::type;
    using bits_type = uint_of<sizeof(type) * byte_size>;
    static constexpr int sign_bits =
        basic_float_traits<custom_double>::sign_bits;
    static constexpr int significand_bits =
        basic_float_traits<custom_double>::significand_bits;
    static constexpr int exponent_bits =
        basic_float_traits<custom_double>::exponent_bits;
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
        basic_float_traits<custom_double>::rounds_to_nearest;

    static constexpr auto eps =
        std::numeric_limits<gko::custom_double>::epsilon();

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
}  // namespace gko::detail


#endif  // GKO_CORE_BASE_DOUBLE_HPP_
