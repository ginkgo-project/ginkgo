// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_INTRINSICS_HPP_
#define GKO_PUBLIC_CORE_BASE_INTRINSICS_HPP_


#include <bitset>

#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace detail {


/**
 * Returns the number of set bits in the given bitmask.
 */
GKO_ATTRIBUTES GKO_INLINE int popcount(uint32 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popc(bitmask);
#else
    std::bitset<32> bits{bitmask};
    return bits.count();
#endif
}


/** @copydoc popcount(uint32) */
GKO_ATTRIBUTES GKO_INLINE int popcount(uint64 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popcll(bitmask);
#else
    std::bitset<64> bits{bitmask};
    return bits.count();
#endif
}


/**
 * Returns the index of the highest bit set in this bitmask.
 * The least significant bit has index 0.
 */
GKO_ATTRIBUTES GKO_INLINE int find_highest_bit(uint32 bitmask)
{
    return 31 -
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
           __clz(bitmask)
#else
           __builtin_clz(bitmask)
#endif
        ;
}


/** @copydoc find_highest_bit(uint32) */
GKO_ATTRIBUTES GKO_INLINE int find_highest_bit(uint64 bitmask)
{
    return 63 -
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
           __clzll(bitmask)
#else
           __builtin_clzll(bitmask)
#endif
        ;
}


/**
 * Returns the index of the lowest bit set in this bitmask.
 * The least significant bit has index 0.
 */
GKO_ATTRIBUTES GKO_INLINE int find_lowest_bit(uint32 bitmask)
{
    return
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        __ffs(bitmask)
#else
        __builtin_ffs(bitmask)
#endif
        - 1;
}


/** @copydoc find_lowest_bit(uint32) */
GKO_ATTRIBUTES GKO_INLINE int find_lowest_bit(uint64 bitmask)
{
    return
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        __ffsll(bitmask)
#else
        __builtin_ffsll(bitmask)
#endif
        - 1;
}


}  // namespace detail
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_INTRINSICS_HPP_
