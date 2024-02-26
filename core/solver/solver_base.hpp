// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_SOLVER_BASE_HPP_
#define GKO_CORE_SOLVER_SOLVER_BASE_HPP_


#include <ginkgo/core/solver/solver_base.hpp>


namespace gko {
namespace solver {


/**
 * Modify the input vector x by the guess
 *
 * @param b  the right hand side vectors
 * @param x  the input vectors
 * @param guess  the input guess
 */
template <typename VectorType>
void prepare_initial_guess(const VectorType* b, VectorType* x,
                           initial_guess_mode guess)
{
    if (guess == initial_guess_mode::zero) {
        x->fill(zero<typename VectorType::value_type>());
    } else if (guess == initial_guess_mode::rhs) {
        x->copy_from(b);
    }
}


}  // namespace solver
}  // namespace gko


#endif  // GKO_CORE_SOLVER_SOLVER_BASE_HPP_
