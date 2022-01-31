/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_REFERENCE_STOP_BATCH_CRITERIA_HPP_
#define GKO_REFERENCE_STOP_BATCH_CRITERIA_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "reference/base/config.hpp"


namespace gko {
namespace kernels {
namespace host {
namespace stop {


/**
 * Stopping criterion for batch solvers that combines a maximum iteration
 * count and relative residual threshold.
 *
 * Supports only one right hand side.
 */
template <typename ValueType>
class SimpleRelResidual {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param max_iters  Maximum number of iterations allowed.
     * @param rel_res_tol  Tolerance on relative residual norm.
     * @param rhs_b_norms  The reference RHS norms.
     */
    SimpleRelResidual(const real_type rel_res_tol,
                      const real_type* const rhs_b_norms)
        : rel_tol_{rel_res_tol}, rhs_norms_{rhs_b_norms}
    {}

    /**
     * Checks whether the right hand side has converged.
     *
     * @param residual_norms  Current residual norm.
     *
     * @return  True if RHS has converged, false otherwise.
     */
    bool check_converged(const real_type* const residual_norms) const
    {
        return (residual_norms[0] / rhs_norms_[0] < rel_tol_);
    }

private:
    const real_type rel_tol_;
    const real_type* const rhs_norms_;
};


/**
 * Stopping criterion for batch solvers that combines a maximum iteration
 * count and absolute residual threshold.
 *
 * Supports only one right hand side.
 */
template <typename ValueType>
class SimpleAbsResidual {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param max_iters  Maximum number of iterations allowed.
     * @param tol  Tolerance on residual norm.
     */
    SimpleAbsResidual(const real_type tol, const real_type*) : abs_tol_{tol} {}

    /**
     * Checks whether the different right hand sides have converged.
     *
     * @param iter  The current iteration count.
     * @param residual_norms  (Optional) current residual norm of each RHS.
     * @param residual  Current residual vectors. Unused if residual_norms
     *                  are provided.
     * @param converged  Bits representing converged (1) or not (0) for each
     *                   RHS. The 'right-most' bit corresponds to the first RHS.
     *
     * @return  True if all RHS have converged, false otherwise.
     */
    bool check_converged(const real_type* const residual_norms) const
    {
        return (residual_norms[0] < abs_tol_);
    }

private:
    const real_type abs_tol_;
};


}  // namespace stop
}  // namespace host
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_STOP_BATCH_CRITERIA_HPP_
