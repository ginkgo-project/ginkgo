// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/sor_kernels.hpp"

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "omp/factorization/factorization_helpers.hpp"

namespace gko {
namespace kernels {
namespace omp {
namespace sor {


template <typename ValueType, typename IndexType>
void initialize_weighted_l(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx)
{
    auto inv_weight = one(weight) / weight;
    factorization::helpers::initialize_l(
        system_matrix, l_mtx,
        factorization::helpers::triangular_mtx_closure(
            [inv_weight](auto val) { return val * inv_weight; },
            [](auto val) { return val; }));
};

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L);


template <typename ValueType, typename IndexType>
void initialize_weighted_l_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* system_matrix,
    remove_complex<ValueType> weight, matrix::Csr<ValueType, IndexType>* l_mtx,
    matrix::Csr<ValueType, IndexType>* u_mtx)
{
    auto inv_weight = one(weight) / weight;
    auto inv_two_minus_weight =
        one(weight) / (static_cast<remove_complex<ValueType>>(2.0) - weight);
    factorization::helpers::initialize_l_u(
        system_matrix, l_mtx, u_mtx,
        factorization::helpers::triangular_mtx_closure(
            [inv_weight](auto val) { return val * inv_weight; },
            [](auto val) { return val; }),
        factorization::helpers::triangular_mtx_closure(
            [inv_two_minus_weight](auto val) {
                return val * inv_two_minus_weight;
            },
            [weight, inv_two_minus_weight](auto val) {
                return val * weight * inv_two_minus_weight;
            }));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SOR_INITIALIZE_WEIGHTED_L_U);


}  // namespace sor
}  // namespace omp
}  // namespace kernels
}  // namespace gko
