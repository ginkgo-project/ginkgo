// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CSR_STRATEGY_HPP_
#define GKO_CORE_MATRIX_CSR_STRATEGY_HPP_

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace matrix {
namespace csr {
namespace detail {


/**
 * Returns the actual strategy passed. When the strategy is automatic, this
 * returns the actual underlying strategy. This returns the same strategy as
 * the input when the input is not automatic.
 *
 * @param exec  Executor associated to the matrix
 * @param strategy  the strategy of CSR
 * @param num_stored_elements  the number of stored elements
 * @param max_nnz_per_row  the maximum number of stored elements per row
 *
 * @return the actual strategy
 */
spmv_strategy get_actual_strategy(std::shared_ptr<const Executor> exec,
                                  spmv_strategy strategy,
                                  size_type num_stored_elements,
                                  size_type max_nnz_per_row);


}  // namespace detail
}  // namespace csr
}  // namespace matrix
}  // namespace gko

#endif  // GKO_CORE_MATRIX_CSR_STRATEGY_HPP_
