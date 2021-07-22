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
    ConstrainedHandler() = default;
    ConstrainedHandler(Array<int32> idxs,
                       std::shared_ptr<const matrix::Dense<double>> values)
    {}

    /**
     * Setups the constrained system.
     *
     * Afterwards, the modified system can be obtained from get_operator,
     * get_right_hand_side, and get_initial_guess. If no initial guess was
     * provided, the guess will be set to zero.
     */
    void setup_system(std::shared_ptr<LinOp> op,
                      std::shared_ptr<const LinOp> rhs,
                      std::shared_ptr<const LinOp> init = nullptr)
    {}

    /**
     * Sets new contrained values, the corresponding indices are not changed.
     *
     * @note Invalidates previous pointers from get_operator,
     * get_right_hand_side, and get_initial_guess
     *
     */
    void update_constrained_values(std::shared_ptr<const matrix::Dense<double>>)
    {}

    /**
     * Set a new operator for the linear system.
     *
     * @note Invalidates previous pointers from get_operator and
     * get_right_hand_side
     *
     * This will also update the right hand side and initial guess.
     * If those should also be changes, than setup_system is better suited.
     */
    void update_operator(std::shared_ptr<LinOp>) {}

    /**
     * Set a new right hand side for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side
     */
    void update_right_hand_side(std::shared_ptr<const LinOp>) {}

    /**
     * Set a new initial guess for the linear system.
     *
     * @note Invalidates previous pointers from get_right_hand_side and
     * get_initial_guess
     *
     * This will also update the right hand side.
     */
    void update_initial_guess(std::shared_ptr<const LinOp>) {}

    /**
     * Read access to the constrained operator
     */
    const LinOp *get_operator() {}

    /**
     * Read access to the right hand side of the constrained system
     */
    const LinOp *get_right_hand_side() {}

    /**
     * Read/write access to the initial guess for the constrained system
     *
     * @note if this function is called multiple times, the initial guess will
     * be rebuild after the the first invocation
     */
    LinOp *get_initial_guess() {}


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
