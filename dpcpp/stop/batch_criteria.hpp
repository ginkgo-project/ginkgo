// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_STOP_BATCH_CRITERIA_HPP_
#define GKO_DPCPP_STOP_BATCH_CRITERIA_HPP_


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/stop/batch_stop_enum.hpp>


#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_stop {


/**
 * @see reference/stop/batch_criteria.hpp
 */
template <typename ValueType>
class SimpleRelResidual {
public:
    using real_type = remove_complex<ValueType>;

    SimpleRelResidual(const real_type rel_res_tol,
                      const real_type* const rhs_b_norms)
        : rel_tol_{rel_res_tol}, rhs_norms_{rhs_b_norms}
    {}

    __dpct_inline__ bool check_converged(
        const real_type* const residual_norms) const
    {
        return residual_norms[0] <= (rel_tol_ * rhs_norms_[0]);
    }

private:
    const real_type rel_tol_;
    const real_type* const rhs_norms_;
};


/**
 * @see reference/stop/batch_criteria.hpp
 */
template <typename ValueType>
class SimpleAbsResidual {
public:
    using real_type = remove_complex<ValueType>;

    SimpleAbsResidual(const real_type tol, const real_type*) : abs_tol_{tol} {}

    __dpct_inline__ bool check_converged(
        const real_type* const residual_norms) const
    {
        return (residual_norms[0] <= abs_tol_);
    }

private:
    const real_type abs_tol_;
};


}  // namespace batch_stop
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_DPCPP_STOP_BATCH_CRITERIA_HPP_
