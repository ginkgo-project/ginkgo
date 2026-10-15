// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_STOP_BATCH_STOP_ENUM_HPP_
#define GKO_PUBLIC_CORE_STOP_BATCH_STOP_ENUM_HPP_


namespace gko {
namespace batch {
namespace stop {


/**
 * This enum provides two types of options for the convergence of an iterative
 * solver.
 *
 * `absolute` tolerance implies that the convergence criteria check is
 * against the computed residual (\f$||r|| \leq \tau\f$)
 *
 * With the `relative` tolerance type, the solver
 * convergence criteria checks against the relative residual norm
 * (\f$||r|| \leq ||b|| \times \tau\f$, where \f$||b||\f$ is the L2 norm of the
 * rhs).
 *
 * @note the computed residual norm, \f$||r||\f$ may be implicit or explicit
 * depending on the solver algorithm.
 */
enum class tolerance_type { absolute, relative };


}  // namespace stop
}  // namespace batch
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_STOP_BATCH_STOP_ENUM_HPP_
