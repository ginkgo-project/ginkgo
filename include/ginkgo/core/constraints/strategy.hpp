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

#ifndef GKO_PUBLIC_CORE_CONSTRAINTS_STRATEGY_HPP_
#define GKO_PUBLIC_CORE_CONSTRAINTS_STRATEGY_HPP_


#include <memory>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/index_set.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace constraints {
namespace detail {


/**
 * Creates a dense vector with fixed values on specific indices and zero
 * elsewhere.
 *
 * @param size  Total size of the vector.
 * @param idxs  The indices that should be set.
 * @param values  The values that should be set.
 */
template <typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
zero_guess_with_constrained_values(std::shared_ptr<const Executor> exec,
                                   dim<2> size, const IndexSet<IndexType>& idxs,
                                   const matrix::Dense<ValueType>* values);


}  // namespace detail


/**
 * Interface for applying constraints to a linear system.
 *
 * This interface provides several methods that are necessary to construct the
 * individual parts of a linear system with constraints, namely:
 * - incorporating the constraints into the operator
 * - deriving a suitable right-hand-side
 * - deriving a suitable initial guess
 * - if necessary, update the solution.
 * Depending on the actual implementation, some of these methods might be
 * no-ops.
 *
 * A specialized implementation of constraints handling can be achieved by
 * deriving a class from this interface and passing it to the
 * ConstraintsHandler.
 *
 * @tparam ValueType The ValueType of the underlying operator and vectors
 * @tparam IndexType The IndexType of the underlying operator and vectors
 */
template <typename ValueType, typename IndexType>
class ApplyConstraintsStrategy
    : public EnableCreateMethod<
          ApplyConstraintsStrategy<ValueType, ValueType>> {
public:
    /**
     * Incorporates the constraints into the operator.
     *
     * @note This might (but not necessarily) change the operator directly.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param op  The original operator.
     * @return  An operator with constraints added.
     */
    virtual std::shared_ptr<LinOp> construct_operator(
        const IndexSet<IndexType>& idxs, std::shared_ptr<LinOp> op) = 0;

    /**
     * Creates a new right-hand-side for the constrained system.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param op  The original (unconstrained) operator.
     * @param init_guess  The original initial guess of the system
     * @param rhs  The original right-hand-side.
     * @return  The right-hand-side for the constrained system.
     */
    virtual std::unique_ptr<LinOp> construct_right_hand_side(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* rhs) = 0;

    /**
     * Creates a new initial guess for the constrained system.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param op  The original (unconstrained) operator.
     * @param init_guess  The original initial guess of the system
     * @param constrained_values  The values of the constrained indices.
     * @return  A new initial guess for the constrained system.
     */
    virtual std::unique_ptr<LinOp> construct_initial_guess(
        const IndexSet<IndexType>& idxs, const LinOp* op,
        const matrix::Dense<ValueType>* init_guess,
        const matrix::Dense<ValueType>* constrained_values) = 0;

    /**
     * If necessary, updates the solution to the constrained system to make it
     * the solution of the original system.
     *
     * @param idxs  The indices where the constraints are applied.
     * @param constrained_values  The values of the constrained indices.
     * @param orig_init_guess The original (unconstrained) initial guess of the
     * system
     * @param solution The solution to the constrained system.
     */
    virtual void correct_solution(
        const IndexSet<IndexType>& idxs,
        const matrix::Dense<ValueType>* constrained_values,
        const matrix::Dense<ValueType>* orig_init_guess,
        matrix::Dense<ValueType>* solution) = 0;
};


}  // namespace constraints
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONSTRAINTS_STRATEGY_HPP_
