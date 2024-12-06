// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/async_jacobi_kernels.hpp"

#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The async_jacobi solver namespace.
 *
 * @ingroup async_jacobi
 */
namespace async_jacobi {


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const std::string& check, int max_iters,
           const matrix::Dense<ValueType>* relaxation_factor,
           const matrix::Dense<ValueType>* second_factor,
           const matrix::Csr<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_ASYNC_JACOBI_APPLY_KERNEL);


}  // namespace async_jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
