// SPDX-FileCopyrightText: 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_BASE_BATCH_INSTANTIATION_HPP_
#define GKO_PUBLIC_CORE_BASE_BATCH_INSTANTIATION_HPP_

#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/matrix/batch_identity.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


namespace gko {
namespace batch {


// just make the call list more consistent
#define GKO_CALL(_macro, ...) _macro(__VA_ARGS__)

#define GKO_BATCH_INSTANTIATE_PRECONDITIONER(_next, ...) \
    _next(__VA_ARGS__, gko::batch::matrix::Identity);    \
    _next(__VA_ARGS__, gko::batch::preconditioner::Jacobi)

#define GKO_BATCH_INSTANTIATE_MATRIX(_next, ...)   \
    _next(__VA_ARGS__, gko::batch::matrix::Ell);   \
    _next(__VA_ARGS__, gko::batch::matrix::Dense); \
    _next(__VA_ARGS__, gko::batch::matrix::Csr)

/**
 * Instantiates a template for each valid combination of value type, batch
 * matrix type, and batch preconditioner type. This only allows batch matrix
 * type and preconditioner type also uses the same value type.
 *
 * @param args   the first should be a macro which expands the template
 *               instantiation (not including the leading `template` specifier).
 *               Should take three arguments, where the first is replaced by the
 *               value type, the second by the matrix, and the third by the
 *               preconditioner.
 *
 * @note the second and third arguments only accept the base type.s
 */
#define GKO_INSTANTIATE_FOR_BATCH_VALUE_MATRIX_PRECONDITIONER(...) \
    GKO_CALL(GKO_BATCH_INSTANTIATE_MATRIX,                         \
             GKO_BATCH_INSTANTIATE_PRECONDITIONER,                 \
             GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_VARGS, __VA_ARGS__)


}  // namespace batch
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_BASE_BATCH_INSTANTIATION_HPP_
