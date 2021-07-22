/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
 */
/*
DPCT1004:0: Compatible DPC++ code could not be generated.
*/
__dpct_inline__ int ffs(uint32 mask) { return __ffs(mask); }

/** @copydoc ffs */
__dpct_inline__ int ffs(uint64 mask)
{
    // the cast is necessary, as the overloads defined by HIP are ambiguous
    /*
    DPCT1004:1: Compatible DPC++ code could not be generated.
    */
    return __ffsll(static_cast<unsigned long long int>(mask));
}


/**
 * @internal
 * Returns the number of zero bits before the first set bit in the given mask,
 * starting from the most significant bit.
 */
__dpct_inline__ int clz(uint32 mask) { return sycl::clz((int)mask); }

/** @copydoc clz */
__dpct_inline__ int clz(uint64 mask) { return sycl::clz((long long)mask); }


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_INTRINSICS_DP_HPP_
