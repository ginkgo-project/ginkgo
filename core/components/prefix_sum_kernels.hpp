/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_COMPONENTS_PREFIX_SUM_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_PREFIX_SUM_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/half.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


/**
 * \fn prefix_sum
 * Computes an exclusive prefix sum or exclusive scan of the input array.
 *
 * As with the standard definition of exclusive scan, the last entry of the
 * input array is not read at all, but is written to.
 * If the input is [3,4,1,9,100], it will be replaced by
 * [0,3,7,8,17].
 * The input values of the prefix sum must be non-negative, and the operation
 * throws OverflowError if one of the additions would overflow.
 *
 * \tparam IndexType  Type of entries to be scanned (summed).
 *
 * \param exec  Executor on which to run the scan operation
 * \param counts  The input/output array to be scanned with the sum operation
 * \param num_entries  Size of the array, equal to one more than the number
 *                     of entries to be summed.
 */
#define GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL(IndexType)                 \
    void prefix_sum_nonnegative(std::shared_ptr<const DefaultExecutor> exec, \
                                IndexType* counts, size_type num_entries)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename IndexType>    \
    GKO_DECLARE_PREFIX_SUM_NONNEGATIVE_KERNEL(IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(components,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko

#endif  // GKO_CORE_COMPONENTS_PREFIX_SUM_KERNELS_HPP_
