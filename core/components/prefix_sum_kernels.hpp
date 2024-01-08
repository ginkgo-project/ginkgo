// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_COMPONENTS_PREFIX_SUM_KERNELS_HPP_
#define GKO_CORE_COMPONENTS_PREFIX_SUM_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
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
