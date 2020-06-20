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

#include "core/solver/multigrid_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <iostream>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The MULTIGRID solver namespace.
 *
 * @ingroup multigrid
 */
namespace multigrid {


template <typename ValueType>
void kcycle_step_1(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Dense<ValueType> *rho,
                   const matrix::Dense<ValueType> *v,
                   matrix::Dense<ValueType> *g, matrix::Dense<ValueType> *d,
                   matrix::Dense<ValueType> *e)
{
    const auto nrows = g->get_size()[0];
    const auto nrhs = g->get_size()[1];
    for (size_type i = 0; i < nrhs; i++) {
        auto temp = alpha->at(0, i) / rho->at(0, i);
        for (size_type j = 0; j < nrows; j++) {
            if (is_finite(temp)) {
                g->at(j, i) -= temp * v->at(j, i);
                e->at(j, i) *= temp;
            }
            d->at(j, i) = e->at(j, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_1_KERNEL);


template <typename ValueType>
void kcycle_step_2(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Dense<ValueType> *rho,
                   const matrix::Dense<ValueType> *gamma,
                   const matrix::Dense<ValueType> *beta,
                   const matrix::Dense<ValueType> *zeta,
                   const matrix::Dense<ValueType> *d,
                   matrix::Dense<ValueType> *e)
{
    const auto nrows = e->get_size()[0];
    const auto nrhs = e->get_size()[1];
    for (size_type i = 0; i < nrhs; i++) {
        auto scaler_d = zeta->at(0, i) /
                        (beta->at(0, i) -
                         gamma->at(0, i) * gamma->at(0, i) / rho->at(0, i));
        auto scaler_e =
            one<ValueType>() - gamma->at(0, i) / alpha->at(0, i) * scaler_d;
        std::cout << scaler_d << " " << scaler_e << std::endl;
        if (is_finite(scaler_d) && is_finite(scaler_e)) {
            for (size_type j = 0; j < nrows; j++) {
                e->at(j, i) = scaler_e * e->at(j, i) + scaler_d * d->at(j, i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MULTIGRID_KCYCLE_STEP_2_KERNEL);


template <typename ValueType>
void kcycle_check_stop(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType> *old_norm,
                       const matrix::Dense<ValueType> *new_norm,
                       const ValueType rel_tol, bool &is_stop)
{
    is_stop = true;
    for (size_type i = 0; i < old_norm->get_size()[1]; i++) {
        if (new_norm->at(0, i) > rel_tol * old_norm->at(0, i)) {
            is_stop = false;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_MULTIGRID_KCYCLE_CHECK_STOP_KERNEL);


}  // namespace multigrid
}  // namespace reference
}  // namespace kernels
}  // namespace gko