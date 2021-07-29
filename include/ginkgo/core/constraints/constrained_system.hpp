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

#ifndef GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_HPP_
#define GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_HPP_

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include <memory>

namespace gko {
namespace constraints {

namespace detail {}


class ApplyConstraintsStrategy
    : public EnableCreateMethod<ApplyConstraintsStrategy> {
public:
    virtual std::unique_ptr<LinOp> construct_operator(const Array<int32>& idxs,
                                                      const LinOp* op) = 0;

    virtual std::unique_ptr<LinOp> construct_right_hand_side(
        const Array<int32>& idxs, const LinOp* op, const LinOp* init_guess,
        const LinOp* rhs) = 0;

    virtual std::unique_ptr<LinOp> construct_initial_guess(
        const Array<int32>& idxs, const LinOp* op,
        const LinOp* constrained_values) = 0;

    virtual void correct_solution(const Array<int32>& idxs,
                                  const LinOp* orig_init_guess,
                                  LinOp* solution) = 0;
};


class ZeroRowsStrategy : public ApplyConstraintsStrategy {};


class ConstrainedHandler {
public:
    /**
     * Initializes the constrained system.
     *
     * Applies the constrains to the system operator and constructs the new
     * initial guess and right hand side using the specified strategy. If no
     * initial guess is supplied, the guess is set to zero.
     *
     * @param idxs  the indices of the constrained degrees of freedom
     * @param system_operator  the original system operator
     * @param values  the values of the constrained defrees of freedom
     * @param right_hand_side  the original right-hand-side of the system
     * @param initial_guess  the initial guess for the original system, if it is
     *                       null, then zero will be used as initial guess
     * @param strategy  the implementation strategy of the constraints
     */
    ConstrainedHandler(
        Array<int32> idxs, std::shared_ptr<const LinOp> system_operator,
        std::shared_ptr<const matrix::Dense<double>> values,
        std::shared_ptr<const matrix::Dense<double>> right_hand_side,
        std::shared_ptr<const matrix::Dense<double>> initial_guess = nullptr,
        std::unique_ptr<ApplyConstraintsStrategy> strategy =
            std::make_unique<ZeroRowsStrategy>())
        : idxs_(std::move(idxs)),
          orig_operator_(std::move(system_operator)),
          values_(std::move(values)),
          orig_rhs_(std::move(right_hand_side)),
          orig_init_guess_(
              initial_guess
                  ? std::move(initial_guess)
                  : share(initialize<matrix::Dense<double>>(
                        0., orig_rhs_->get_size(), orig_rhs_->get_executor()))),
          strategy_(std::move(strategy))
    {
        GKO_ASSERT(values && right_hand_side);

        cons_operator_ =
            strategy->construct_operator(idxs_, lend(orig_operator_));
        if (orig_init_guess_ && values_) {
            cons_init_guess_ = strategy_->construct_initial_guess(
                idxs, lend(cons_operator_), lend(values_));
        }
        if (orig_rhs_ && cons_init_guess_) {
            cons_rhs_ = strategy_->construct_right_hand_side(
                idxs, lend(cons_operator_), lend(cons_init_guess_),
                lend(values_));
        }
    }

    /**
     * Initializes the constrained system.
     *
     * Applies the constrains to the system operator using the specified
     * strategy. The constrained values, right-hand-side, and initial guess have
     * to be set later on via the with_* functions.
     *
     * @param idxs  the indices of the constrained degrees of freedom
     * @param system_operator  the original system operator
     * @param strategy  the implementation strategy of the constraints
     */
    ConstrainedHandler(Array<int32> idxs,
                       std::shared_ptr<LinOp> system_operator,
                       std::unique_ptr<ApplyConstraintsStrategy> strategy =
                           std::make_unique<ZeroRowsStrategy>())
        : ConstrainedHandler(std::move(idxs), std::move(system_operator),
                             nullptr, nullptr, nullptr, std::move(strategy))
    {}

    /**
     * Sets new contrained values, the corresponding indices are not changed.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * @return *this
     */
    ConstrainedHandler& with_constrained_values(
        std::shared_ptr<const matrix::Dense<double>> values)
    {
        values_ = std::move(values);

        // invalidate previous pointers
        cons_init_guess_.reset();
        cons_rhs_.reset();

        return *this;
    }

    /**
     * Set a new right hand side for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side
     *
     * @return *this
     */
    ConstrainedHandler& with_right_hand_side(
        std::shared_ptr<const LinOp> right_hand_side)
    {
        orig_rhs_ = std::move(right_hand_side);

        // invalidate previous pointer
        cons_rhs_.reset();

        return *this;
    }

    /**
     * Set a new initial guess for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * @return *this
     */
    ConstrainedHandler& with_initial_guess(
        std::shared_ptr<const LinOp> initial_guess)
    {
        orig_init_guess_ = std::move(initial_guess);

        // invalidate previous pointers
        cons_init_guess_.reset();
        cons_rhs_.reset();

        return *this;
    }

    /**
     * Read access to the constrained operator
     */
    const LinOp* get_operator() { return cons_operator_.get(); }

    /**
     * Read access to the right hand side of the constrained system.
     *
     * First call after with_right_hand_side, with_initial_guess, or
     * with_constrained_values constructs the constrained right-hand-side and
     * initial guess if necessary. Without further with_* calls, this function
     * does not recompute the right-hand-side.
     */
    const LinOp* get_right_hand_side()
    {
        if (!cons_rhs_) {
            if (!cons_init_guess_) {
                cons_init_guess_ = strategy_->construct_initial_guess(
                    idxs_, lend(cons_operator_), lend(values_));
            }
            cons_rhs_ = strategy_->construct_right_hand_side(
                idxs_, lend(cons_operator_), lend(cons_init_guess_),
                lend(values_));
        }
        return cons_rhs_.get();
    }

    /**
     * Read/write access to the initial guess for the constrained system
     *
     * Without providing an initial guess either to the constructor or
     * with_initial_guess, zero will be assumed for the initial guess of the
     * original system.
     *
     * @note Reconstructs the initial guess at every call.
     */
    LinOp* get_initial_guess()
    {
        if (!cons_init_guess_) {
            cons_init_guess_ = strategy_->construct_initial_guess(
                idxs_, lend(cons_operator_), lend(values_));
        }
        return cons_init_guess_.get();
    }

    /**
     * Forces the reconstruction of the constrained system.
     *
     * Afterwards, the modified system can be obtained from get_operator,
     * get_right_hand_side, and get_initial_guess. If no initial guess was
     * provided, the guess will be set to zero.
     */
    void reconstruct_system()
    {
        cons_init_guess_ = strategy_->construct_initial_guess(
            idxs_, lend(cons_operator_), lend(values_));
        cons_rhs_ = strategy_->construct_right_hand_side(
            idxs_, lend(cons_operator_), lend(cons_init_guess_), lend(values_));
    }

    /**
     * Obtains the solution to the original constrained system from the solution
     * of the modified system
     */
    void correct_solution(LinOp* solution)
    {
        strategy_->correct_solution(idxs_, lend(orig_init_guess_), solution);
    }

private:
    Array<int32> idxs_;

    std::shared_ptr<const LinOp> orig_operator_;
    std::unique_ptr<LinOp> cons_operator_;

    std::unique_ptr<ApplyConstraintsStrategy> strategy_;

    std::shared_ptr<const matrix::Dense<double>> values_;

    std::shared_ptr<const LinOp> orig_rhs_;
    std::unique_ptr<LinOp> cons_rhs_;
    std::shared_ptr<const LinOp> orig_init_guess_;
    std::unique_ptr<LinOp> cons_init_guess_;
};


}  // namespace constraints
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_HPP_
