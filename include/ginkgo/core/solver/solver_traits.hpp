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
