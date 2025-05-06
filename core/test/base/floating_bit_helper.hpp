// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_BASE_FLOATING_BIT_HELPER_HPP_
#define GKO_CORE_TEST_BASE_FLOATING_BIT_HELPER_HPP_


#include <bitset>
#include <cstring>


namespace floating_bit_helper {


constexpr auto byte_size = gko::detail::byte_size;


template <typename T, std::size_t N>
T create_from_bits(const char (&s)[N])
{
    auto bits = std::bitset<N - 1>(s).to_ullong();
    // We cast to the same size of integer type first.
    // Otherwise, the first memory chunk is different when we use
    // reinterpret_cast or memcpy to get the smaller type out of unsigned
    // long long.
    using bits_type = typename gko::detail::float_traits<T>::bits_type;
    auto bits_val = static_cast<bits_type>(bits);
    T result;
    static_assert(sizeof(T) == sizeof(bits_type),
                  "the type should have the same size as its bits_type");
    std::memcpy(&result, &bits_val, sizeof(bits_type));
    return result;
}


template <typename T>
std::bitset<sizeof(T) * byte_size> get_bits(T val)
{
    using bits_type = typename gko::detail::float_traits<T>::bits_type;
    bits_type bits;
    static_assert(sizeof(T) == sizeof(bits_type),
                  "the type should have the same size as its bits_type");
    std::memcpy(&bits, &val, sizeof(T));
    return std::bitset<sizeof(T) * byte_size>(bits);
}

template <std::size_t N>
std::bitset<N - 1> get_bits(const char (&s)[N])
{
    return std::bitset<N - 1>(s);
}


}  // namespace floating_bit_helper

#endif  // GKO_CORE_TEST_BASE_FLOATING_BIT_HELPER_HPP_
