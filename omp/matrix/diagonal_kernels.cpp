// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/diagonal_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Diagonal<ValueType>* a,
                  const matrix::Csr<ValueType, IndexType>* b,
                  matrix::Csr<ValueType, IndexType>* c, bool inverse)
{
    const auto diag_values = a->get_const_values();
    c->copy_from(b);
    auto csr_values = c->get_values();
    const auto csr_row_ptrs = c->get_const_row_ptrs();

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; row++) {
        const auto scal =
            inverse ? one<ValueType>() / diag_values[row] : diag_values[row];
        for (size_type idx = csr_row_ptrs[row]; idx < csr_row_ptrs[row + 1];
             idx++) {
            csr_values[idx] *= scal;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


}  // namespace diagonal
}  // namespace omp
}  // namespace kernels
}  // namespace gko
