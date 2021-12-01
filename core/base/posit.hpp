#pragma once
#include <algorithm>
#include <bitset>
#include <cinttypes>
#include <climits>  // for CHAR_BIT (number of bits in a byte)
#include <cmath>
#include <cstring>
#include <iomanip>
#include <ios>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>


#if defined(_MSC_VER)
#include <intrin.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define POSIT_ATTRIBUTES __host__ __device__
#else
#define POSIT_ATTRIBUTES
#endif


namespace detail {


template <typename T>
struct ieee_details {};

template <>
struct ieee_details<float> {
    using ieee_type = double;
    using fp_uint_type = std::uint32_t;
    static constexpr int exponent_bits{8};
    static constexpr int significand_bits{23};
};

template <>
struct ieee_details<double> {
    using ieee_type = double;
    using fp_uint_type = std::uint64_t;
    static constexpr int exponent_bits{11};
    static constexpr int significand_bits{52};
};


#if defined(__CUDACC__) || defined(__HIPCC__)


__device__ double uint_to_ieee(std::uint64_t uint_value)
{
    return __longlong_as_double(uint_value);
}

__device__ float uint_to_ieee(std::uint32_t uint_value)
{
    return __uint_as_float(uint_value);
}


__device__ std::uint64_t ieee_to_uint(double fp_value)
{
    return __double_as_longlong(fp_value);
}

__device__ std::uint32_t ieee_to_uint(float fp_value)
{
    return __float_as_uint(fp_value);
}


__device__ int countl_zero(std::uint64_t val) { return __clzll(val); }

__device__ int countl_zero(std::uint32_t val) { return __clz(val); }

__device__ int countl_zero(std::uint16_t val)
{
    return val == 0 ? 16 : countl_zero(static_cast<std::uint32_t>(val) << 16);
}


__device__ int countl_one(std::uint64_t val) { return countl_zero(~val); }

__device__ int countl_one(std::uint32_t val) { return countl_zero(~val); }

__device__ int countl_one(std::uint16_t val)
{
    return val == static_cast<std::uint16_t>(~std::uint16_t{0})
               ? 16
               : countl_one(static_cast<std::uint32_t>(val) << 16);
}


#else  // Host compilation


#if defined(_MSC_VER)


int countl_zero(std::uint64_t val)
{
    static_assert(sizeof(std::uint64_t) <= sizeof(unsigned long long),
                  "Argument type must fit into builtin function!");
    unsigned long ret;
    return _BitScanForward(&ret, val) == 0 ? 64 : 63 - static_cast<int>(ret);
}

int countl_zero(std::uint32_t val)
{
    static_assert(sizeof(std::uint32_t) <= sizeof(unsigned long),
                  "Argument type must fit into builtin function!");
    unsigned long ret;
    return _BitScanForward(&ret, val) == 0 ? 32 : 31 - static_cast<int>(ret);
}

int countl_zero(std::uint16_t val)
{
    return val == 0 ? 16 : countl_zero(static_cast<std::uint32_t>(val) << 16);
}


int countl_one(std::uint64_t val) { return countl_zero(~val); }

int countl_one(std::uint32_t val) { return countl_zero(~val); }

int countl_one(std::uint16_t val)
{
    return val == static_cast<std::uint16_t>(~std::uint16_t{0})
               ? 16
               : countl_one(static_cast<std::uint32_t>(val) << 16);
}


#elif defined(__GNUC__) || defined(__clang__)


int countl_zero(std::uint64_t val)
{
    static_assert(std::is_same<std::uint64_t, unsigned long>::value,
                  "Types must have the same size!");
    return val == 0 ? 64 : __builtin_clzl(val);
}

int countl_zero(std::uint32_t val)
{
    static_assert(std::is_same<std::uint32_t, unsigned int>::value,
                  "Types must have the same size!");
    return val == 0 ? 32 : __builtin_clz(val);
}

int countl_zero(std::uint16_t val)
{
    return val == 0 ? 16 : countl_zero(static_cast<std::uint32_t>(val) << 16);
}


int countl_one(std::uint64_t val) { return countl_zero(~val); }

int countl_one(std::uint32_t val) { return countl_zero(~val); }

int countl_one(std::uint16_t val)
{
    static_assert(
        std::is_same<std::int32_t, decltype(~std::uint16_t{0})>::value,
        "Must Match!!!");
    return val == static_cast<std::uint16_t>(~std::uint16_t{0})
               ? 16
               : countl_one(static_cast<std::uint32_t>(val) << 16);
}


#else  // Neither MSC for GNU for clang


template <typename T>
std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value, int>
countl_zero(T val)
{
    if (val == 0) {
        return sizeof(T) * CHAR_BIT;
    }
    constexpr T boundary{T{1} << (sizeof(T) * CHAR_BIT - 1)};
    int i{0};
    for (; val < boundary; ++i) {
        val <<= 1;
    }
    return i;
}


template <typename T>
std::enable_if_t<std::is_integral<T>::value && !std::is_signed<T>::value, int>
countl_one(T val)
{
    if (val == 0) {
        return sizeof(T) * CHAR_BIT;
    }
    constexpr T boundary{T{1} << (sizeof(T) * CHAR_BIT - 1)};
    int i{0};
    for (; val >= boundary; ++i) {
        val <<= 1;
    }
    return i;
}


#endif  // End of default implementation
#endif  // End of host section


template <typename T>
struct ieee_helper {
    static_assert(std::numeric_limits<T>::is_iec559,
                  "The type must adhere to the IEC 559 (IEEE 754) Standard!");
    using ieee_type = T;
    using fp_uint_type = typename ieee_details<T>::fp_uint_type;
    static constexpr int number_bits{sizeof(T) * CHAR_BIT};
    static constexpr auto exponent_bits = ieee_details<T>::exponent_bits;
    static constexpr auto significand_bits = ieee_details<T>::significand_bits;

    static_assert(1 + exponent_bits + significand_bits == number_bits,
                  "Number of bits does not add up!");
    static_assert(sizeof(ieee_type) == sizeof(fp_uint_type),
                  "Proxy type must match IEEE type in size!");

    static constexpr int exponent_bias{
        (fp_uint_type{1} << (exponent_bits - 1)) - 1};

    static constexpr fp_uint_type sign_mask{fp_uint_type{1}
                                            << (number_bits - 1)};
    static constexpr fp_uint_type exponent_mask{
        ((fp_uint_type{1} << exponent_bits) - 1) << significand_bits};
    static constexpr fp_uint_type significand_mask{
        (fp_uint_type{1} << significand_bits) - 1};

    static POSIT_ATTRIBUTES fp_uint_type to_uint(const ieee_type fp_value)
    {
#if defined(__CUDA_ARCH__)
        return ieee_to_uint(fp_value);
#else
        fp_uint_type ui{};
        std::memcpy(&ui, &fp_value, sizeof(fp_uint_type));
        return ui;
#endif
    }

    static POSIT_ATTRIBUTES ieee_type to_ieee(const fp_uint_type uint_value)
    {
#if defined(__CUDA_ARCH__)
        return uint_to_ieee(uint_value);
#else
        ieee_type fp{};
        std::memcpy(&fp, &uint_value, sizeof(fp_uint_type));
        return fp;
#endif
    }
};


template <int number_bits, typename verify = void*>
struct best_int {
    static_assert(number_bits < 0, "Negative values not supported!");
    static_assert(std::is_same<verify, void*>::value,
                  "Do not touch the verify parameter!");
};

template <int number_bits>
struct best_int<number_bits,
                std::enable_if_t<0 <= number_bits && number_bits <= 8>*> {
    using type = std::uint8_t;
};

template <int number_bits>
struct best_int<number_bits,
                std::enable_if_t<8 < number_bits && number_bits <= 16>*> {
    using type = std::uint16_t;
};

template <int number_bits>
struct best_int<number_bits,
                std::enable_if_t<16 < number_bits && number_bits <= 32>*> {
    using type = std::uint32_t;
};

template <int number_bits>
struct best_int<number_bits,
                std::enable_if_t<32 < number_bits && number_bits <= 64>*> {
    using type = std::uint64_t;
};


template <int number_bits>
using best_int_t = typename best_int<number_bits>::type;


template <int number_bits, typename Uint>
constexpr std::enable_if_t<number_bits == sizeof(Uint) * CHAR_BIT, Uint>
clear_fill_bits(Uint input)
{
    return input;
}

template <int number_bits, typename Uint>
constexpr std::enable_if_t<(number_bits < sizeof(Uint) * CHAR_BIT), Uint>
clear_fill_bits(Uint input)
{
    return input & ((Uint{1} << (number_bits - 1)) - 1);
}


template <int number_bits, typename Uint>
constexpr std::enable_if_t<number_bits == sizeof(Uint) * CHAR_BIT, Uint>
set_fill_bits(Uint input)
{
    return input;
}

template <int number_bits, typename Uint>
constexpr std::enable_if_t<(number_bits < sizeof(Uint) * CHAR_BIT), Uint>
set_fill_bits(Uint input)
{
    return input | ~((Uint{1} << (number_bits - 1)) - 1);
}


}  // namespace detail


template <int NumberBits, int ES>
struct posit {
public:
    static_assert(0 < NumberBits, "Number of bits must be at least 1!");
    static_assert(NumberBits <= 64, "Number of bits must be 64 or lower!");
    static_assert(0 <= ES, "ES must be at least 0!");
    static_assert(ES < NumberBits, "ES must be lower than NumberBits!");

    // Maximum number of exponent bits
    static constexpr int es{ES};
    static constexpr int number_bits{NumberBits};

    using uint_proxy_type = detail::best_int_t<number_bits>;
    // log_2 of useed ( = 2^es) )
    static constexpr int two_power_es{es == 0 ? 1 : (2 << (es - 1))};
    // Base value for the regime ( = 2^(2^es)) )
    static constexpr int useed{1 << two_power_es};

    struct dissection {
        bool sign;
        int number_r_bits;
        int k;
        int number_e_bits;
        int e;
        int number_f_bits;

        friend std::ostream& operator<<(std::ostream& out,
                                        const dissection& dis)
        {
            out << 's' << dis.sign << '(' << (dis.sign ? '-' : '+') << ')';
            out << ' ' << dis.k << '(' << dis.number_r_bits << ')' << ';';
            out << dis.e << '(' << dis.number_e_bits << ')' << ';';
            out << '(' << dis.number_f_bits << ')';
            return out;
        }
    };

    // This is the only non-static and non-constexpr attribute
    uint_proxy_type memory;

    constexpr bool get_sign() const { return memory & sign_mask_; }

    POSIT_ATTRIBUTES dissection dissect() const
    {
        dissection dis{};

        // If the special case of NaN is reached
        if (this->memory == sign_mask_) {
            dis.sign = true;
            dis.number_r_bits = number_bits - 1;
            dis.k = -dis.number_r_bits;
            return dis;
        } else if (this->memory == 0) {
            dis.sign = false;
            dis.number_r_bits = number_bits - 1;
            dis.k = -dis.number_r_bits;
            return dis;
        }

        dis.sign = get_sign();
        auto mem = this->memory;
        // if it is a negative number, the rest is stored in the 2's complement
        if (dis.sign) {
            mem = twos_complement(mem);
        }

        extract_regime(mem, dis);

        // Extract exponent
        // `es` needs to be made into a temporary, so it does not need to be
        // de-referenced
        dis.number_e_bits =
            std::min(int{es}, number_bits - 1 - dis.number_r_bits);
        auto exp =
            mem >> (number_bits - 1 - dis.number_r_bits - dis.number_e_bits);
        if (dis.number_e_bits > 0) {
            dis.e = exp & ((uint_proxy_type{1} << dis.number_e_bits) - 1);
            dis.e <<= es - dis.number_e_bits;
        }

        // Extract fraction
        dis.number_f_bits =
            number_bits - 1 - dis.number_r_bits - dis.number_e_bits;

        return dis;
    }

    template <typename IeeeType>
    POSIT_ATTRIBUTES IeeeType to_ieee() const
    {
        using ieee_type = IeeeType;
        using ieee = detail::ieee_helper<ieee_type>;
        using fp_uint_type = typename ieee::fp_uint_type;

        if (memory == 0) {
            return 0;
        } else if (memory == sign_mask_) {
            return std::numeric_limits<ieee_type>::quiet_NaN();
        }

        auto dis = this->dissect();
        fp_uint_type ui_val{};
        // Set sign bit
        if (dis.sign) {
            ui_val |= fp_uint_type{1} << (sizeof(fp_uint_type) * CHAR_BIT - 1);
        }

        // Compute Exponent
        // Since useed = 2^(2^es) and the number is computed as
        // sign * useed^k * 2^exp * fraction
        // We can compute the base 2 exponent as:
        // sign * 2^(exp + k * 2^es) * fraction
        const int base_two_exp = dis.e + dis.k * two_power_es;

        // Check for underflow
        if (base_two_exp <= -ieee::exponent_bias) {
            // if denormals are not enough, set it to the smallest value > 0
            if (base_two_exp <
                -ieee::exponent_bias + 1 - ieee::significand_bits + 1) {
                return dis.sign ? -std::numeric_limits<ieee_type>::denorm_min()
                                : std::numeric_limits<ieee_type>::denorm_min();
            } else {
                // denormal is enough: compute the additional shift of the
                // significand
                const int additional_right_shifts =
                    -base_two_exp - ieee::exponent_bias;

                // Make the implicit 1 in front of the fraction explicit
                // Also, for negative numbers, take the two's complement
                const auto posit_fraction =
                    ((dis.sign ? twos_complement(this->memory) : this->memory) &
                     ((uint_proxy_type{1} << dis.number_f_bits) - 1)) |
                    (uint_proxy_type{1} << dis.number_f_bits);
                // Set the proper significand
                ui_val |= right_shift<fp_uint_type>(
                    posit_fraction,
                    dis.number_f_bits + 1 -
                        (ieee::significand_bits - additional_right_shifts));
            }
        } else if (base_two_exp > ieee::exponent_bias) {
            return dis.sign ? -std::numeric_limits<ieee_type>::max()
                            : std::numeric_limits<ieee_type>::max();
        } else {
            const auto ieee_exp = base_two_exp + ieee::exponent_bias;
            ui_val |= static_cast<fp_uint_type>(ieee_exp)
                      << ieee::significand_bits;

            // For negative numbers, take the two's complement
            const auto posit_fraction =
                (dis.sign ? twos_complement(this->memory) : this->memory) &
                ((uint_proxy_type{1} << dis.number_f_bits) - 1);
            ui_val |= right_shift<fp_uint_type>(
                posit_fraction, dis.number_f_bits - ieee::significand_bits);
        }

        // copy bits to IEEE value
        return ieee::to_ieee(ui_val);
    }

    POSIT_ATTRIBUTES void set_k_e(const int exponent, dissection& dis)
    {
        auto k_e = compute_k_e(exponent);
        dis.k = std::get<0>(k_e);
        dis.e = std::get<1>(k_e);

        uint_proxy_type regime{};
        // Somewhat underflow, assign the lowest number
        if (dis.k <= -(number_bits - 1)) {
            dis.number_r_bits = number_bits - 1;
            dis.k = -dis.number_r_bits;
            dis.e = 0;
            // Smallest regime possible, therefore, regime is all 0s except the
            // least significand bit (otherwise, it would be the special case
            // memory == 0)
            regime = uint_proxy_type{1};
        }
        // Somewhat overflow, assign the highest number
        else if (dis.k >= number_bits - 2) {
            dis.number_r_bits = number_bits - 1;
            dis.k = dis.number_r_bits - 1;
            dis.e = 0;
            // Largest regime possible, therefore, regime is all 1s
            regime = (uint_proxy_type{1} << (dis.number_r_bits)) - 1;
        } else {
            if (dis.k < 0) {
                dis.number_r_bits = -dis.k + 1;
                regime = 1;
            } else {
                dis.number_r_bits = dis.k + 2;
                regime = (uint_proxy_type{1} << (dis.number_r_bits)) - 2;
            }
            // Move the regime to the correct place
            regime <<= number_bits - 1 - dis.number_r_bits;
        }
        // Update memory with the regime
        this->memory |= regime;

        // Figure out how many bits the exponent has, clean it up and move
        // it to memory
        dis.number_e_bits =
            std::min(int{es}, number_bits - 1 - dis.number_r_bits);
        dis.number_f_bits =
            number_bits - 1 - dis.number_r_bits - dis.number_e_bits;
        uint_proxy_type posit_exponent{static_cast<uint_proxy_type>(dis.e)};

        // If we don't have all expnent bits, right shifts are needed
        // Note: e and k, as well as the number of bits will not be properly
        //       updated
        if (dis.number_e_bits < es) {
            this->memory |= posit_exponent >> (es - dis.number_e_bits);
            if ((posit_exponent >> (es - dis.number_e_bits - 1)) & 1) {
                // Round up
                this->memory += 1;
            }
        } else {
            this->memory |= posit_exponent << dis.number_f_bits;
        }
    }

    template <typename T>
    POSIT_ATTRIBUTES void from_ieee(T val)
    {
        using ieee_type = T;
        using ieee = detail::ieee_helper<ieee_type>;
        using fp_uint_type = typename ieee::fp_uint_type;

        dissection dis;
        this->memory = 0;  // reset memory

        // Check for NaN or infinity
        if (!std::isfinite(val)) {
            this->memory = sign_mask_;
            return;
        }
        // Make sure -0 is also set to 0
        else if (val == T{0}) {
            return;
        }

        const auto ui_val = ieee::to_uint(val);

        const int exponent = static_cast<int>((ui_val & ieee::exponent_mask) >>
                                              ieee::significand_bits) -
                             ieee::exponent_bias;
        // Detect denormals
        if (exponent == -ieee::exponent_bias) {
            int exponent_correction{1};
            // TODO improve implementation
            auto current_significand = ui_val & ieee::significand_mask;
            while (!(current_significand &
                     (fp_uint_type{1}
                      << (ieee::significand_bits - exponent_correction))) &&
                   exponent_correction < ieee::significand_bits) {
                ++exponent_correction;
            }
            // Remove the first `1` as it will be an implicit 1 in the POSIT
            // format
            current_significand = current_significand ^
                                  (fp_uint_type{1} << (ieee::significand_bits -
                                                       exponent_correction));
            // New exponent according to IEEE denormals uses bias
            // `exponent_bias - 1`
            const auto new_exponent =
                -ieee::exponent_bias + 1 - exponent_correction;

            set_k_e(new_exponent, dis);

            // Set fraction accordingly
            this->memory |= right_shift<uint_proxy_type>(
                current_significand, ieee::significand_bits -
                                         exponent_correction -
                                         dis.number_f_bits);
        } else {
            set_k_e(exponent, dis);

            const auto num_right_shifts =
                ieee::significand_bits - dis.number_f_bits;

            this->memory |= right_shift<uint_proxy_type>(
                ui_val & ieee::significand_mask, num_right_shifts);

            // Rounding (Here, we only consider positive numbers as the sign is
            // only applied at the very end)

            // If we right shift, we lose precision -> rounding
            if (num_right_shifts > 0 && dis.number_f_bits > 0) {
                // Rounding implementation (mostly) according to
                // https://www.posithub.org/posit_standard4.12.pdf
                auto cutoff =
                    ui_val & ((fp_uint_type{1} << num_right_shifts) - 1);
                if (cutoff < (fp_uint_type{1} << (num_right_shifts - 1))) {
                    // Round down ^= leave it cut off
                }
                // Special case (cutoff == 100000...) makes no sense in the
                // documentation because they don't account for negative
                // numbers, therefore, we skip it
                else {
                    // Round up
                    this->memory += 1;
                }
            }
        }
        // Do the sign at the end, so the two's complement can be done in
        // one go
        bool sign = static_cast<bool>(ui_val & ieee::sign_mask);
        if (sign) {
            memory =
                detail::clear_fill_bits<number_bits>(twos_complement(memory));
        }
    }

    constexpr posit operator-() const
    {
        // Since 0 and NaR are preserved in the two's complement, zeroing out
        // the filling bits is all we need to do to preserve binary comparisons
        return {detail::clear_fill_bits<number_bits>(
            twos_complement(this->memory))};
    }

    constexpr bool operator==(const posit& other) const
    {
        return this->memory == other.memory;
    }

    POSIT_ATTRIBUTES operator double() const
    {
        return this->template to_ieee<double>();
    }

    POSIT_ATTRIBUTES operator float() const
    {
        return this->template to_ieee<float>();
    }

    constexpr posit() = default;

    constexpr posit(uint_proxy_type val) : memory{val} {}

    POSIT_ATTRIBUTES posit(double val) { this->from_ieee(val); }

    POSIT_ATTRIBUTES posit(float val) { this->from_ieee(val); }


private:
    static constexpr uint_proxy_type sign_mask_{uint_proxy_type{1}
                                                << (number_bits - 1)};
    static constexpr uint_proxy_type msb_regime_mask_{uint_proxy_type{1}
                                                      << (number_bits - 2)};

    // TODO: Figure out what to do if e would be cut off and is larger than the
    //       number of bits (in practice, it needs to be shifted to the right
    //       since the missing bits are assumed to be zero)
    static constexpr std::tuple<int, int> compute_k_e(int base_two_exponent)
    {
        int k = base_two_exponent / two_power_es;
        const int e = base_two_exponent % two_power_es;
        // Since the e exponent is only positive, we need to round
        // towards negative infinity when computing k
        if (base_two_exponent < 0 &&
            static_cast<bool>(base_two_exponent & (two_power_es - 1))) {
            k = k - 1;
        }
        // TODO Fix negative number results for e (in case base_two_exponent is
        // negative) properly
        return {k, (e < 0 ? two_power_es + e : e)};
    }

    // Compute the two's complement of a given number by using unary minus.
    // C++11 expr.unary.op #8 (5.3.1 #7) for details
    static constexpr uint_proxy_type twos_complement(uint_proxy_type val)
    {
// Disable unary minus for unsigned number warning for Visual Studio
#ifdef _MSC_VER
#pragma warning(disable : 4146)
#endif
        return -val;
#ifdef _MSC_VER
#pragma warning(default : 4146)
#endif
    }

    // right_shifts: number of right shifts to adopt the given input to
    // the OutputType. Can be negative to indicate left shift.
    template <typename OutputType, typename UintType>
    static POSIT_ATTRIBUTES OutputType right_shift(const UintType input,
                                                   const int right_shifts)
    {
        if (right_shifts > 0) {
            return static_cast<OutputType>(input >> (right_shifts));
        } else {
            return static_cast<OutputType>(input) << (-right_shifts);
        }
    }

    static POSIT_ATTRIBUTES void extract_regime(uint_proxy_type mem,
                                                dissection& dis)
    {
        // Count for the regime; Needs to be optimized!
        // Filling bits + sign bit are the leading non-regime bits
        constexpr int leading_non_regime_bits =
            1 + sizeof(uint_proxy_type) * CHAR_BIT - number_bits;
        const bool regime_sign = static_cast<bool>(mem & msb_regime_mask_);
        if (!regime_sign) {
            // Additional cast necessary because of potential integer promotion
            const auto num_zeros =
                detail::countl_zero(static_cast<uint_proxy_type>(
                    detail::clear_fill_bits<number_bits>(mem))) -
                leading_non_regime_bits;
            dis.k = -(num_zeros);
            dis.number_r_bits = std::min(num_zeros + 1, number_bits - 1);
        } else {
            // Additional cast necessary because of potential integer promotion
            const auto num_ones =
                detail::countl_one(static_cast<uint_proxy_type>(
                    detail::set_fill_bits<number_bits>(mem) | sign_mask_)) -
                leading_non_regime_bits;
            dis.k = num_ones - 1;
            dis.number_r_bits = std::min(num_ones + 1, number_bits - 1);
        }
    }
};


template <int NB, int ES>
std::ostream& operator<<(std::ostream& out, const posit<NB, ES>& pos)
{
    // Save current flags and settings
    const auto out_flags = out.flags();
    const auto old_prec = out.precision();

    out << std::noshowpos;

    auto dis = pos.dissect();
    auto fp64 = pos.template to_ieee<double>();

    out << "Posit<" << NB << ',' << ES << ">: ";

    std::bitset<NB> pos_bs(pos.memory);
    const auto start_sign = NB - 1;
    const auto start_regime = start_sign - dis.number_r_bits;
    const auto start_exponent = dis.number_f_bits;
    for (int i = NB - 1; i >= 0; --i) {
        if (i == start_sign - 1 || i == start_regime - 1 ||
            i == start_exponent - 1) {
            out << ' ';
        }
        out << static_cast<int>(pos_bs.test(i));
    }

    out << "  k: " << dis.k << " (" << dis.number_r_bits << "); e: " << dis.e
        << " (" << dis.number_e_bits << "); f: (" << dis.number_f_bits << ");";

    out << std::scientific << std::showpos;

    out << " to fp64: " << std::setprecision(18) << fp64;

    // Restore settings and flags for out
    out.flags(out_flags);
    out.precision(old_prec);

    return out;
}

// According to the POSIT standard at:
// https://posithub.org/posit_standard4.12.pdf
using posit16_2 = posit<16, 2>;
using posit32_2 = posit<32, 2>;
using posit32_3 = posit<32, 3>;
