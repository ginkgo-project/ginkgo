// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_COMPONENTS_MATRIX_OPERATIONS_DP_HPP_
#define GKO_DPCPP_COMPONENTS_MATRIX_OPERATIONS_DP_HPP_


#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {


/**
 * @internal
 *
 * Computes the infinity norm of a column-major matrix.
 */
template <typename ValueType>
remove_complex<ValueType> compute_inf_norm(
    size_type num_rows, size_type num_cols, const ValueType* matrix,
    size_type stride) GKO_NOT_IMPLEMENTED;


}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_COMPONENTS_MATRIX_OPERATIONS_DP_HPP_
