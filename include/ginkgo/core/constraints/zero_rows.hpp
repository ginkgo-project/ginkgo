/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_PUBLIC_CORE_CONSTRAINTS_ZERO_ROWS_HPP_
#define GKO_PUBLIC_CORE_CONSTRAINTS_ZERO_ROWS_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/constraints/strategy.hpp>
#include <ginkgo/core/matrix/dense.hpp>

namespace gko {
namespace constraints {


/**
 * Applies constraints to a linear system, by zeroing-out rows.
 *
 * The rows of a matrix that correspond to constrained values are set to zero,
 * except the diagonal entry, which is set to 1. This directly changes the
 * values of the matrix, and the operator's symmetry is not maintained. However,
 * the operator is still symmetric (or self-adjoint) on a subspace, where the
 * constrained indices of vectors are set to zero, so that the constrained
 * operator can still be used in a CG method for example. Additionally, a new
 * right-hand-side in that subspace is created as `new_b = b - cons_A * x_0` and
 * the new initial guess is set to 0 for constrained indices. Lastly, the
 * constrained values are added to the solution of the system `cons_A * z =
 * new_b`.
 *
 * @note Current restrictions:
 * - can only be used with a single right-hand-side
 * - can only be used with `stride=1` vectors
 *
 * @tparam ValueType The ValueType of the underlying operator and vectors
 * @tparam IndexType The IndexType of the underlying operator and vectors
 */
template <typename ValueType, typename IndexType>
class ZeroRowsStrategy : public ApplyConstraintsStrategy<ValueType, IndexType> {
    using Dense = matrix::Dense<ValueType>;

public:
    std::shared_ptr<LinOp> construct_operator(
        const IndexSet<IndexType>& idxs, std::shared_ptr<LinOp> op) override;

    std::unique_ptr<LinOp> construct_right_hand_side(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* rhs) override;

    std::unique_ptr<LinOp> construct_initial_guess(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* constrained_values) override;

    void correct_solution(const IndexSet<IndexType>& idxs,
                          const matrix::Dense<ValueType>* constrained_values,
                          const matrix::Dense<ValueType>* orig_init_guess,
                          matrix::Dense<ValueType>* solution) override;

private:
    std::unique_ptr<Dense> one;
    std::unique_ptr<Dense> neg_one;
};


}  // namespace constraints
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONSTRAINTS_ZERO_ROWS_HPP_
