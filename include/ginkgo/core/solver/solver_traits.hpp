// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_PUBLIC_CORE_SOLVER_SOLVER_TRAITS_HPP_
#define GKO_PUBLIC_CORE_SOLVER_SOLVER_TRAITS_HPP_


#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/stop/criterion.hpp>


namespace gko {
namespace solver {


/**
 * Helper structure to test if the Factory of SolverType has a function
 * `with_criteria`.
 *
 * Contains a constexpr boolean `value`, which is true if the Factory class
 * of SolverType has a `with_criteria`, and false otherwise.
 *
 * @tparam SolverType   Solver to test if its factory has a with_criteria
 *                      function.
 *
 */
template <typename SolverType, typename = void>
struct has_with_criteria : std::false_type {};

/**
 * @copydoc has_with_criteria
 *
 * @internal  The second template parameter (which uses SFINAE) must match
 *            the default value of the general case in order to be accepted
 *            as a specialization, which is why `xstd::void_t` is used.
 */
template <typename SolverType>
struct has_with_criteria<
    SolverType, xstd::void_t<decltype(SolverType::build().with_criteria(
                    std::shared_ptr<const stop::CriterionFactory>()))>>
    : std::true_type {};


}  // namespace solver
}  // namespace gko

#endif  // GKO_PUBLIC_CORE_SOLVER_SOLVER_TRAITS_HPP_
