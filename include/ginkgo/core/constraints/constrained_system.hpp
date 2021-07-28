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
#include <ginkgo/core/matrix/dense.hpp>

#include <memory>

namespace gko {
namespace constraints {

class ConstrainedHandler {
public:
    /**
     * Initializes the constrained system.
     *
     * Applies the constrains to the system operator.
     * Other parts of the original system may be passed as well, but their
     * constrained versions are not constructed until either get_* or
     * construct_system is called.
     * If these parts are not passed, they must be set with the respected with_*
     * call.
     *
     * @param idxs  the indices of the constrained degrees of freedom
     * @param system_operator  the original system operator
     * @param values  the values of the constrained defrees of freedom
     * @param right_hand_side  the original right-hand-side of the system
     * @param initial_guess  the initial guess for the original system
     */
    ConstrainedHandler(
        Array<int32> idxs, std::shared_ptr<LinOp> system_operator,
        std::shared_ptr<const matrix::Dense<Value>> values = nullptr,
        std::shared_ptr<const matrix::Dense<Value>> right_hand_side = nullptr,
        std::shared_ptr<const matrix::Dense<Value>> initial_guess = nullptr)
    {}

    /**
     * Sets new contrained values, the corresponding indices are not changed.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * @return *this
     */
    ConstrainedHandler &with_constrained_values(
        std::shared_ptr<const matrix::Dense<double>>)
    {}

    /**
     * Set a new right hand side for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side
     *
     * @return *this
     */
    ConstrainedHandler &with_right_hand_side(std::shared_ptr<const LinOp>) {}

    /**
     * Set a new initial guess for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * @return *this
     */
    ConstrainedHandler &with_initial_guess(std::shared_ptr<const LinOp>) {}

    /**
     * Read access to the constrained operator
     */
    const LinOp *get_operator() {}

    /**
     * Read access to the right hand side of the constrained system.
     *
     * First call after with_right_hand_side, with_initial_guess, or
     * with_constrained_values constructs the constrained right-hand-side.
     * Without further with_* calls, this function does not recompute the
     * right-hand-side.
     */
    const LinOp *get_right_hand_side() {}

    /**
     * Read/write access to the initial guess for the constrained system
     *
     * Without providing an initial guess either to the constructor or
     * with_initial_guess, zero will be assumed for the initial guess of the
     * original system.
     *
     * @note Reconstructs the initial guess at every call.
     */
    LinOp *get_initial_guess() {}

    /**
     * Forces the construction of the constrained system.
     *
     * Afterwards, the modified system can be obtained from get_operator,
     * get_right_hand_side, and get_initial_guess. If no initial guess was
     * provided, the guess will be set to zero.
     */
    void construct_system() {}

    /**
     * Obtains the solution to the original constrained system from the solution
     * of the modified system
     */
    void correct_solution(LinOp *) {}

private:
    Array<int32> idxs_;
    std::shared_ptr<const matrix::Dense<double>> values_;

    std::shared_ptr<const LinOp> orig_operator_;
    std::unique<LinOp> cons_operator_;

    std::shared<const LinOp> orig_rhs_;
    std::unique_ptr<LinOp> cons_rhs_;
    std::shared<const LinOp> orig_init_guess_;
    std::unique_ptr<LinOp> cons_init_guess_;
};


}  // namespace constraints
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_CONSTRAINTS_CONSTRAINED_SYSTEM_HPP_
