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

/**
 * Instantiates a template for each valid combination of value type, batch
 * matrix type, and batch preconditioner type. This only allows batch matrix
 * type and preconditioner type also uses the same value type.
 *
 * @param _macro  A macro which expands the template instantiation
 *                (not including the leading `template` specifier).
 *                Should take three arguments, where the first is replaced by
 *                the value type, the second by the matrix, and the third by the
 *                preconditioner.
 *
 * @note the second and third arguments only accept the base type.s
 */
#define GKO_INSTANTIATE_FOR_BATCH_VALUE_MATRIX_PRECONDITIONER(_macro)         \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_ARGS(_macro, gko::batch::matrix::Csr, \
                                             gko::batch::matrix::Identity);   \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_ARGS(_macro, gko::batch::matrix::Ell, \
                                             gko::batch::matrix::Identity);   \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_ARGS(                                 \
        _macro, gko::batch::matrix::Dense, gko::batch::matrix::Identity);     \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_ARGS(                                 \
        _macro, gko::batch::matrix::Csr, gko::batch::preconditioner::Jacobi); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_ARGS(                                 \
        _macro, gko::batch::matrix::Ell, gko::batch::preconditioner::Jacobi); \
    GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_ARGS(                                 \
        _macro, gko::batch::matrix::Dense, gko::batch::preconditioner::Jacobi)

}  // namespace batch
}  // namespace gko

#endif  //
