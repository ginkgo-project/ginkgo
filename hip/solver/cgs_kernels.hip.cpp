/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#include "core/solver/cgs_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The CGS solver namespace.
 *
 * @ingroup cgs
 */
namespace cgs {


template <typename ValueType>
void initialize(std::shared_ptr<const HipExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r_tld, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *u,
                matrix::Dense<ValueType> *u_hat,
                matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *beta,
                matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *rho_prev,
                matrix::Dense<ValueType> *rho,
                Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *rho_prev,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *t, matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *gamma,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const HipExecutor> exec,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *u_hat, matrix::Dense<ValueType> *r,
            matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *alpha,
            const Array<stopping_status> *stop_status) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace hip
}  // namespace kernels
}  // namespace gko
