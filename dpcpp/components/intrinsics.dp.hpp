// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_INTRINSICS_DP_HPP_
#define GKO_DPCPP_COMPONENTS_INTRINSICS_DP_HPP_


#include <CL/sycl.hpp>


#include <ginkgo/core/base/types.hpp>


#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * @internal
 * Returns the number of set bits in the given mask.
 */
__dpct_inline__ int popcnt(uint32 mask) { return sycl::popcount(mask); }

/** @copydoc popcnt */
__dpct_inline__ int popcnt(uint64 mask) { return sycl::popcount(mask); }


/**
 * @internal
 * Returns the (1-based!) index of the first set bit in the given mask,
 * starting from the least significant bit.
 *
 * @note can not use length - clz because subgroup size is not always 32 or 64
 */
__dpct_inline__ int ffs(uint32 mask)
{
    return (mask == 0) ? 0 : (sycl::ext::intel::ctz(mask) + 1);
}

/** @copydoc ffs */
__dpct_inline__ int ffs(uint64 mask)
{
    return (mask == 0) ? 0 : (sycl::ext::intel::ctz(mask) + 1);
}


/**
 * @internal
 * Returns the number of zero bits before the first set bit in the given mask,
 * starting from the most significant bit.
 */
__dpct_inline__ int clz(uint32 mask) { return sycl::clz(mask); }

/** @copydoc clz */
__dpct_inline__ int clz(uint64 mask) { return sycl::clz(mask); }


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_INTRINSICS_DP_HPP_
