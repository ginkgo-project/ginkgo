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

#include <ginkgo/core/solver/solver_base.hpp>


#include <type_traits>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/utils_helper.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace solver {


template <typename ValueType>
void ApplyWithInitialGuess::prepare_initial_guess(
    const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x,
    initial_guess_mode guess)
{
    if (guess == initial_guess_mode::zero) {
        x->fill(zero<value_type>());
    } else if (guess == initial_guess_mode::rhs) {
        x->copy_from(b);
    }
}

#define GKO_DECLARE_APPLY_WITH_INITIAL_GUESS_PREPARE_INITIAL_GUESS(_type) \
    void ApplyWithInitialGuess::prepare_initial_guess(                    \
        const matrix::Dense<_type>*, matrix::Dense<_type>*,               \
        initial_guess_mode)
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_APPLY_WITH_INITIAL_GUESS_PREPARE_INITIAL_GUESS);


void ApplyWithInitialGuess::apply_with_initial_guess(const LinOp* b, LinOp* x,
                                                     initial_guess_mode guess)
{
    if (guess == initial_guess_mode::zero) {
        run<matrix::Dense<float>*, matrix::Dense<double>*,
            matrix::Dense<std::complex<float>>*,
            matrix::Dense<std::complex<double>>*>(x, [&](auto dense) {
            using value_type = typename std::decay_t<
                gko::detail::pointee<decltype(dense)>>::value_type;
            dense->fill(zero<value_type>());
        });
    } else if (guess == initial_guess_mode::rhs) {
        x->copy_from(b);
    }
    this->apply_impl(b, x);
}


void ApplyWithInitialGuess::apply_with_initial_guess(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x,
    initial_guess_mode guess) const
{
    if (guess == initial_guess_mode::zero) {
        run<matrix::Dense<float>*, matrix::Dense<double>*,
            matrix::Dense<std::complex<float>>*,
            matrix::Dense<std::complex<double>>*>(x, [&](auto dense) {
            using value_type = typename std::decay_t<
                gko::detail::pointee<decltype(dense)>>::value_type;
            dense->fill(zero<value_type>());
        });
    } else if (guess == initial_guess_mode::rhs) {
        x->copy_from(b);
    }
    this->apply_impl(alpha, b, beta, x);
}


}  // namespace solver
}  // namespace gko
