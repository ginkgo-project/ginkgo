/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/solver/cgs_kernels.hpp"


#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace cgs {


template <typename ValueType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r_tld, matrix::Dense<ValueType> *p,
                matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *u,
                matrix::Dense<ValueType> *u_hat,
                matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *alpha, matrix::Dense<ValueType> *beta,
                matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho)
{
    for (size_type j = 0; j < b->get_num_cols(); ++j) {
        rho->at(j) = zero<ValueType>();
        prev_rho->at(j) = one<ValueType>();
        alpha->at(j) = one<ValueType>();
        beta->at(j) = one<ValueType>();
        gamma->at(j) = one<ValueType>();
    }
    for (size_type i = 0; i < b->get_num_rows(); ++i) {
        for (size_type j = 0; j < b->get_num_cols(); ++j) {
            r->at(i, j) = b->at(i, j);
            r_tld->at(i, j) = b->at(i, j);
            u->at(i, j) = u_hat->at(i, j) = p->at(i, j) = q->at(i, j) =
                v_hat->at(i, j) = t->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p)
{
    for (size_type i = 0; i < p->get_num_rows(); ++i) {
        for (size_type j = 0; j < p->get_num_cols(); ++j) {
            u->at(i, j) = r->at(i, j);
            p->at(i, j) = r->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *u,
            matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *beta, const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *rho_prev)
{
    for (size_type j = 0; j < p->get_num_cols(); ++j) {
        if (rho_prev->at(j) != zero<ValueType>()) {
            beta->at(j) = rho->at(j) / rho_prev->at(j);
        }
    }
    for (size_type i = 0; i < p->get_num_rows(); ++i) {
        for (size_type j = 0; j < p->get_num_cols(); ++j) {
            u->at(i, j) = r->at(i, j) + beta->at(j) * q->at(i, j);
            p->at(i, j) =
                u->at(i, j) +
                beta->at(j) * (q->at(i, j) + beta->at(j) * p->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);

template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *u,
            const matrix::Dense<ValueType> *v_hat, matrix::Dense<ValueType> *q,
            matrix::Dense<ValueType> *t, matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *gamma)
{
    for (size_type j = 0; j < u->get_num_cols(); ++j) {
        if (gamma->at(j) != zero<ValueType>()) {
            alpha->at(j) = rho->at(j) / gamma->at(j);
        }
    }
    for (size_type i = 0; i < u->get_num_rows(); ++i) {
        for (size_type j = 0; j < u->get_num_cols(); ++j) {
            q->at(i, j) = u->at(i, j) - alpha->at(j) * v_hat->at(i, j);
            t->at(i, j) = u->at(i, j) + q->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);

template <typename ValueType>
void step_4(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *u_hat, matrix::Dense<ValueType> *r,
            matrix::Dense<ValueType> *x, const matrix::Dense<ValueType> *alpha)
{
    for (size_type i = 0; i < x->get_num_rows(); ++i) {
        for (size_type j = 0; j < x->get_num_cols(); ++j) {
            x->at(i, j) += alpha->at(j) * u_hat->at(i, j);
            r->at(i, j) -= alpha->at(j) * t->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_4_KERNEL);


}  // namespace cgs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
