// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_COMPONENTS_MATRIX_OPERATIONS_HPP_
#define GKO_REFERENCE_COMPONENTS_MATRIX_OPERATIONS_HPP_


#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace reference {


/**
 * @internal
 *
 * Computes the infinity norm of a column-major matrix.
 */
template <typename ValueType>
remove_complex<ValueType> compute_inf_norm(size_type num_rows,
                                           size_type num_cols,
                                           const ValueType* matrix,
                                           size_type stride)
{
    auto result = zero<remove_complex<ValueType>>();
    for (size_type i = 0; i < num_rows; ++i) {
        auto tmp = zero<remove_complex<ValueType>>();
        for (size_type j = 0; j < num_cols; ++j) {
            tmp += abs(matrix[i + j * stride]);
        }
        result = max(result, tmp);
    }
    return result;
}


}  // namespace reference
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_COMPONENTS_MATRIX_OPERATIONS_HPP_
