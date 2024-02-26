// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_REFERENCE_STOP_BATCH_CRITERIA_HPP_
#define GKO_REFERENCE_STOP_BATCH_CRITERIA_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


namespace gko {
namespace kernels {
namespace host {
namespace batch_stop {


/**
 * Stopping criterion for batch solvers with relative residual threshold.
 *
 * @note Supports only one right hand side.
 */
template <typename ValueType>
class SimpleRelResidual {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param rel_res_tol  Tolerance on relative residual norm.
     * @param rhs_b_norms  The RHS norms.
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
     * @return  true if converged, false otherwise.
     */
    bool check_converged(const real_type* const residual_norms) const
    {
        return residual_norms[0] <= (rel_tol_ * rhs_norms_[0]);
    }

private:
    const real_type rel_tol_;
    const real_type* const rhs_norms_;
};


/**
 * Stopping criterion for batch solvers that checks for an absolute residual
 * threshold.
 *
 * @note Supports only one right hand side.
 */
template <typename ValueType>
class SimpleAbsResidual {
public:
    using real_type = remove_complex<ValueType>;

    /**
     * Set up the stopping criterion and convergence variable.
     *
     * @param tol  Tolerance on residual norm.
     * @param dummy  for uniform creation of stopping criteria.
     */
    SimpleAbsResidual(const real_type tol, const real_type*) : abs_tol_{tol} {}

    /**
     * Checks whether the different right hand sides have converged.
     *
     * @param residual_norms  current residual norm of each RHS.
     * @return  true if converged, false otherwise.
     */
    bool check_converged(const real_type* const residual_norms) const
    {
        return (residual_norms[0] <= abs_tol_);
    }

private:
    const real_type abs_tol_;
};


}  // namespace batch_stop
}  // namespace host
}  // namespace kernels
}  // namespace gko


#endif  // GKO_REFERENCE_STOP_BATCH_CRITERIA_HPP_
