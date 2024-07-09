// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace sor {


template <typename ValueType, typename IndexType>
void initialize_weighted_l(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight,
    matrix::Csr<ValueType, IndexType>* l_mtx) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L);


template <typename ValueType, typename IndexType>
void initialize_weighted_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx,
    matrix::Csr<ValueType, IndexType>* u_mtx) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U);


}  // namespace sor
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
