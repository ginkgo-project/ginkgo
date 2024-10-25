// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/ic_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The ic factorization namespace.
 *
 * @ingroup factor
 */
namespace ic_factorization {


template <typename ValueType, typename IndexType>
void sparselib_ic(std::shared_ptr<const DefaultExecutor> exec,
                  matrix::Csr<ValueType, IndexType>* m) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_WITH_HALF(
    GKO_DECLARE_IC_SPARSELIB_IC_KERNEL);


}  // namespace ic_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
