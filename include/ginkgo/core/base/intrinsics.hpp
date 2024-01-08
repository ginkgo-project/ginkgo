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


/**
 * Returns the number of set bits in the given bitmask.
 */
GKO_ATTRIBUTES GKO_INLINE int popcount(uint64 bitmask)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __popcll(bitmask);
#else
    std::bitset<64> bits{bitmask};
    return bits.count();
#endif
}


}  // namespace detail
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_INTRINSICS_HPP_
