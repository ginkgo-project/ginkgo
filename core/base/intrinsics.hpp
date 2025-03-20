// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_INTRINSICS_HPP_
#define GKO_CORE_BASE_INTRINSICS_HPP_

#include <bitset>

#include <ginkgo/core/base/types.hpp>

// MSVC needs different intrinsics
#ifdef _MSC_VER
#include <intrin.h>

#pragma intrinsic(_BitScanForward, _BitScanForward64, _BitScanReverse, \
                  _BitScanReverse64)
#endif


namespace gko {
namespace detail {


/**
 * Returns the index of the highest bit set in this bitmask.
 * The least significant bit has index 0.
 */
GKO_ATTRIBUTES GKO_INLINE int find_highest_bit(uint32 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return 31 - __clz(static_cast<unsigned>(bitmask));
#elif defined(_MSC_VER)
    unsigned long index{};
    _BitScanReverse(&index, bitmask);
    return index;
#else
    return 31 - __builtin_clz(bitmask);
#endif
}


/** @copydoc find_highest_bit(uint32) */
GKO_ATTRIBUTES GKO_INLINE int find_highest_bit(uint64 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return 63 - __clzll(static_cast<unsigned long long>(bitmask));
#elif defined(_MSC_VER)
    unsigned long index{};
    _BitScanReverse64(&index, bitmask);
    return index;
#else
    return 63 - __builtin_clzll(bitmask);
#endif
}


/**
 * Returns the index of the lowest bit set in this bitmask.
 * The least significant bit has index 0.
 */
GKO_ATTRIBUTES GKO_INLINE int find_lowest_bit(uint32 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __ffs(static_cast<unsigned>(bitmask)) - 1;
#elif defined(_MSC_VER)
    unsigned long index{};
    _BitScanForward(&index, bitmask);
    return index;
#else
    return __builtin_ffs(bitmask) - 1;
#endif
}


/** @copydoc find_lowest_bit(uint32) */
GKO_ATTRIBUTES GKO_INLINE int find_lowest_bit(uint64 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __ffsll(static_cast<unsigned long long>(bitmask)) - 1;
#elif defined(_MSC_VER)
    unsigned long index{};
    _BitScanForward64(&index, bitmask);
    return index;
#else
    return __builtin_ffsll(bitmask) - 1;
#endif
}


/**
 * Returns the number of set bits in the given bitmask.
 */
GKO_ATTRIBUTES GKO_INLINE int popcount(uint32 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popc(static_cast<unsigned>(bitmask));
#else
    std::bitset<32> bits{bitmask};
    return bits.count();
#endif
}


/** @copydoc popcount(uint32) */
GKO_ATTRIBUTES GKO_INLINE int popcount(uint64 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popcll(static_cast<unsigned long long>(bitmask));
#else
    std::bitset<64> bits{bitmask};
    return bits.count();
#endif
}


}  // namespace detail
}  // namespace gko

#endif  // GKO_CORE_BASE_INTRINSICS_HPP_
