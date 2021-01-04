/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include <ginkgo/core/log/convergence.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/stop/criterion.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


namespace gko {
namespace log {


template <typename ValueType>
void Convergence<ValueType>::on_criterion_check_completed(
    const stop::Criterion *criterion, const size_type &num_iterations,
    const LinOp *residual, const LinOp *residual_norm, const LinOp *solution,
    const uint8 &stopping_id, const bool &set_finalized,
    const Array<stopping_status> *status, const bool &oneChanged,
    const bool &converged) const
{
    if (converged) {
        this->num_iterations_ = num_iterations;
        if (residual != nullptr) {
            this->residual_.reset(residual->clone().release());
        }
        if (residual_norm != nullptr) {
            this->residual_norm_.reset(residual_norm->clone().release());
        } else if (residual != nullptr) {
            using Vector = matrix::Dense<ValueType>;
            using NormVector = matrix::Dense<remove_complex<ValueType>>;
            this->residual_norm_ = NormVector::create(
                residual->get_executor(), dim<2>{1, residual->get_size()[1]});
            auto dense_r = as<Vector>(residual);
            dense_r->compute_norm2(this->residual_norm_.get());
        }
    }
}


#define GKO_DECLARE_CONVERGENCE(_type) class Convergence<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONVERGENCE);


}  // namespace log
}  // namespace gko
