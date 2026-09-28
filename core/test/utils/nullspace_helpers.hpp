// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_TEST_UTILS_NULLSPACE_HELPERS_HPP_
#define GKO_CORE_TEST_UTILS_NULLSPACE_HELPERS_HPP_


#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/nullspace.hpp>


namespace gko {
namespace test {


/**
 * Computes how far `op` is from annihilating the nullspace `ns`: the largest
 * \( \|A q\|_2 \) over the orthonormal basis vectors \( q \) of `ns`, including
 * the normalized constant if it is part of the nullspace. A NaN norm makes the
 * result NaN.
 *
 * @param op  the operator whose (right) nullspace `ns` should be
 * @param ns  the nullspace
 * @param like  a single-column vector (matrix::Dense or
 *              experimental::distributed::Vector) with the layout of the
 *              columns of `op`, which is used for the constant vector
 *
 * @return the largest residual norm
 */
template <typename ValueType, typename VectorType>
remove_complex<ValueType> compute_nullspace_residual(
    const LinOp* op, const solver::Nullspace<ValueType>* ns,
    const VectorType* like)
{
    using real_type = remove_complex<ValueType>;
    auto exec = like->get_executor();
    auto result = zero<real_type>();
    auto update = [&](const VectorType* vectors) {
        auto op_vectors = gko::clone(vectors);
        op->apply(vectors, op_vectors);
        auto norms = matrix::Dense<real_type>::create(
            exec, dim<2>{1, vectors->get_size()[1]});
        op_vectors->compute_norm2(norms);
        auto host_norms = gko::clone(exec->get_master(), norms);
        for (size_type j = 0; j < host_norms->get_size()[1]; ++j) {
            const auto norm = host_norms->at(0, j);
            if (!(norm <= result)) {
                result = norm;
            }
        }
    };
    if (ns->contains_constant()) {
        auto constant = gko::clone(like);
        constant->fill(static_cast<ValueType>(
            one<real_type>() /
            sqrt(static_cast<real_type>(like->get_size()[0]))));
        update(constant.get());
    }
    if (auto basis = ns->get_basis()) {
        update(as<VectorType>(basis.get()));
    }
    return result;
}


}  // namespace test
}  // namespace gko


#endif  // GKO_CORE_TEST_UTILS_NULLSPACE_HELPERS_HPP_
