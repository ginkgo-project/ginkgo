// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_TYPES_HPP_
#define GKO_CORE_BASE_TYPES_HPP_


#include <array>
#include <cstdint>
#include <type_traits>


namespace gko {
namespace detail {


/**
 * mask gives the integer with Size activated bits in the end
 *
 * @tparam Size  the number of activated bits
 * @tparam ValueType  the type of mask, which uses std::uint32_t as default
 *
 * @return the ValueType with Size activated bits in the end
 */
template <int Size, typename ValueType = std::uint32_t>
constexpr std::enable_if_t<(Size < sizeof(ValueType) * 8), ValueType> mask()
{
    return (ValueType{1} << Size) - 1;
}

/**
 * @copydoc mask()
 *
 * @note this is special case for the Size = the number of bits of ValueType
 */
template <int Size, typename ValueType = std::uint32_t>
constexpr std::enable_if_t<Size == sizeof(ValueType) * 8, ValueType> mask()
{
    return ~ValueType{};
}


/**
 * shift calculates the number of bits for shifting
 *
 * @tparam current_shift  the current position of shifting
 * @tparam num_groups  the number of elements in array
 *
 * @return the number of shifting bits
 *
 * @note this is the last case of nested template
 */
template <int current_shift, int num_groups>
constexpr std::enable_if_t<(num_groups == current_shift + 1), int> shift(
    const std::array<unsigned char, num_groups>& bits)
{
    return 0;
}

/**
 * @copydoc shift(const std::array<char, num_groups>)
 *
 * @note this is the usual case of nested template
 */
template <int current_shift, int num_groups>
constexpr std::enable_if_t<(num_groups > current_shift + 1), int> shift(
    const std::array<unsigned char, num_groups>& bits)
{
    return bits[current_shift + 1] +
           shift<(current_shift + 1), num_groups>(bits);
}


}  // namespace detail


/**
 * ConfigSet is a way to embed several information into one integer by given
 * certain bits.
 *
 * The usage will be the following
 * Set the method with bits Cfg = ConfigSet<b_0, b_1, ..., b_k>
 * Encode the given information encoded = Cfg::encode(x_0, x_1, ..., x_k)
 * Decode the specific position information x_t = Cfg::decode<t>(encoded)
 * The encoded result will use 32 bits to record
 * rrrrr0..01....1...k..k, which 1/2/.../k means the bits store the information
 * for 1/2/.../k position and r is for rest of unused bits.
 *
 * Denote $B_t = \sum_{i = t+1}^k b_i$ and $F(X) = Cfg::encode(x_0, ..., x_k)$.
 * Have $F(X) = \sum_{i = 0}^k (x_i << B_i) = \sum_{i = 0}^k (x_i * 2^{B_i})$.
 * For all i, we have $0 <= x_i < 2^{b_i}$.
 * $x_i$, $2^{B_i}$ are non-negative, so
 * $F(X) = 0$ <=> $X = \{0\}$, $x_i = 0$ for all i.
 * Assume $F(X) = F(Y)$, then
 * $0 = |F(X) - F(Y)| = |F(X-Y)| = F(|X - Y|)$.
 * $|x_i - y_i|$ is still in the same range $0 <= |x_i - y_i| < 2^{b_i}$.
 * Thus, $F(|X - Y|) = 0$ -> $|X - Y| = \{0\}$, $x_i - y_i = 0$ -> $X = Y$.
 * F is one-to-one function if $0 <= x_i < 2^{b_i}$ for all i.
 * For any encoded result R, we can use the following to get the decoded series.
 * for i = k to 0;
 *   $x_i = R % b_i$;
 *   $R = R / bi$;
 * endfor;
 * For any R in the range $[0, 2^{B_0})$, we have X such that $F(X) = R$.
 * F is onto function.
 * Thus, F is bijection.
 *
 * @tparam num_bits...  the number of bits for each position.
 *
 * @note the num_bit is required at least $ceil(log_2(maxval) + 1)$
 */
template <unsigned char... num_bits>
class ConfigSet {
public:
    static constexpr unsigned num_groups = sizeof...(num_bits);
    static constexpr std::array<unsigned char, num_groups> bits{num_bits...};

    /**
     * Decodes the `position` information from encoded
     *
     * @tparam position  the position of desired information
     *
     * @param encoded  the encoded integer
     *
     * @return the decoded information at position
     */
    template <int position>
    static constexpr std::uint32_t decode(std::uint32_t encoded)
    {
        static_assert(position < num_groups,
                      "This position is over the bounds.");
        constexpr int shift = detail::shift<position, num_groups>(bits);
        constexpr auto mask = detail::mask<bits[position]>();
        return (encoded >> shift) & mask;
    }

    /**
     * Encodes the information with given bit set to encoded integer.
     *
     * @note the last case of nested template.
     */
    template <unsigned current_iter>
    static constexpr std::enable_if_t<(current_iter == num_groups),
                                      std::uint32_t>
    encode()
    {
        return 0;
    }

    /**
     * Encodes the information with given bit set to encoded integer.
     *
     * @tparam current_iter  the encoded place
     * @tparam Rest...  the rest type
     *
     * @param first  the current encoded information
     * @param rest...  the rest of other information waiting for encoding
     *
     * @return the encoded integer
     */
    template <unsigned current_iter = 0, typename... Rest>
    static constexpr std::enable_if_t<(current_iter < num_groups),
                                      std::uint32_t>
    encode(std::uint32_t first, Rest&&... rest)
    {
        constexpr int shift = detail::shift<current_iter, num_groups>(bits);
        if (current_iter == 0) {
            static_assert(
                bits[current_iter] + shift <= sizeof(std::uint32_t) * 8,
                "the total bits usage is larger than std::uint32_t bits");
        }
        return (first << shift) |
               encode<current_iter + 1>(std::forward<Rest>(rest)...);
    }
};


}  // namespace gko

#endif  // GKO_CORE_BASE_TYPES_HPP_
