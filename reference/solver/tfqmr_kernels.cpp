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

#include "core/solver/tfqmr_kernels.hpp"

#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace tfqmr {


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *r0, matrix::Dense<ValueType> *u_m,
                matrix::Dense<ValueType> *u_mp1, matrix::Dense<ValueType> *pu_m,
                matrix::Dense<ValueType> *Au, matrix::Dense<ValueType> *Ad,
                matrix::Dense<ValueType> *w, matrix::Dense<ValueType> *v,
                matrix::Dense<ValueType> *d, matrix::Dense<ValueType> *taut,
                matrix::Dense<ValueType> *rho_old,
                matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *alpha,
                matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *tau,
                matrix::Dense<ValueType> *sigma, matrix::Dense<ValueType> *rov,
                matrix::Dense<ValueType> *eta, matrix::Dense<ValueType> *nomw,
                matrix::Dense<ValueType> *theta)
{
    for (size_type j = 0; j < b->get_num_cols(); ++j) {
        taut->at(j) = one<ValueType>();
        rho_old->at(j) = zero<ValueType>();
        rho->at(j) = zero<ValueType>();
        alpha->at(j) = zero<ValueType>();
        beta->at(j) = zero<ValueType>();
        tau->at(j) = zero<ValueType>();
        sigma->at(j) = zero<ValueType>();
        rov->at(j) = zero<ValueType>();
        eta->at(j) = one<ValueType>();
        nomw->at(j) = one<ValueType>();
        theta->at(j) = zero<ValueType>();
    }
    for (size_type i = 0; i < b->get_num_rows(); ++i) {
        for (size_type j = 0; j < b->get_num_cols(); ++j) {
            r->at(i, j) = b->at(i, j);
            r0->at(i, j) = b->at(i, j);
            u_m->at(i, j) = b->at(i, j);
            w->at(i, j) = b->at(i, j);
            v->at(i, j) = b->at(i, j);
            d->at(i, j) = zero<ValueType>();
            u_mp1->at(i, j) = zero<ValueType>();
            pu_m->at(i, j) = zero<ValueType>();
            Au->at(i, j) = zero<ValueType>();
            Ad->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *rov,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *u_m,
            matrix::Dense<ValueType> *u_mp1)
{
    for (size_type j = 0; j < v->get_num_cols(); ++j) {
        if (rov->at(j) != zero<ValueType>()) {
            alpha->at(j) = rho->at(j) / rov->at(j);
        } else {
            alpha->at(j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < v->get_num_rows(); ++i) {
        for (size_type j = 0; j < v->get_num_cols(); ++j) {
            u_mp1->at(i, j) = u_m->at(i, j) - alpha->at(j) * v->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *theta,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *eta,
            matrix::Dense<ValueType> *sigma, const matrix::Dense<ValueType> *Au,
            const matrix::Dense<ValueType> *pu_m, matrix::Dense<ValueType> *w,
            matrix::Dense<ValueType> *d, matrix::Dense<ValueType> *Ad)
{
    for (size_type j = 0; j < d->get_num_cols(); ++j) {
        if (alpha->at(j) != zero<ValueType>()) {
            sigma->at(j) = theta->at(j) / alpha->at(j) * eta->at(j);
        } else {
            sigma->at(j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < d->get_num_rows(); ++i) {
        for (size_type j = 0; j < d->get_num_cols(); ++j) {
            w->at(i, j) = w->at(i, j) - alpha->at(j) * Au->at(i, j);
            d->at(i, j) = pu_m->at(i, j) + sigma->at(j) * d->at(i, j);
            Ad->at(i, j) = Au->at(i, j) + sigma->at(j) * Ad->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType> *theta,
            const matrix::Dense<ValueType> *nomw,
            matrix::Dense<ValueType> *taut, matrix::Dense<ValueType> *eta,
            const matrix::Dense<ValueType> *alpha)
{
    for (size_type j = 0; j < alpha->get_num_cols(); ++j) {
        if (taut->at(j) != zero<ValueType>()) {
            theta->at(j) = nomw->at(j) / taut->at(j);
        } else {
            theta->at(j) = zero<ValueType>();
        }
        auto tmp = one<ValueType>() / sqrt(one<ValueType>() + theta->at(j));
        taut->at(j) = taut->at(j) * sqrt(theta->at(j)) * tmp;
        eta->at(j) = alpha->at(j) * tmp * tmp;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_3_KERNEL);


template <typename ValueType>
void step_4(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *eta,
            const matrix::Dense<ValueType> *d,
            const matrix::Dense<ValueType> *Ad, matrix::Dense<ValueType> *x,
            matrix::Dense<ValueType> *r)
{
    for (size_type i = 0; i < d->get_num_rows(); ++i) {
        for (size_type j = 0; j < d->get_num_cols(); ++j) {
            x->at(i, j) = x->at(i, j) + eta->at(j) * d->at(i, j);
            r->at(i, j) = r->at(i, j) - eta->at(j) * Ad->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_4_KERNEL);


template <typename ValueType>
void step_5(std::shared_ptr<const DefaultExecutor> exec,
            matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *rho_old,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *w,
            const matrix::Dense<ValueType> *u_m,
            matrix::Dense<ValueType> *u_mp1)
{
    for (size_type j = 0; j < w->get_num_cols(); ++j) {
        if (rho_old->at(j) != zero<ValueType>()) {
            beta->at(j) = rho->at(j) / rho_old->at(j);
        } else {
            beta->at(j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < w->get_num_rows(); ++i) {
        for (size_type j = 0; j < w->get_num_cols(); ++j) {
            u_mp1->at(i, j) = w->at(i, j) + beta->at(j) * u_m->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_5_KERNEL);

template <typename ValueType>
void step_6(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *Au_new,
            const matrix::Dense<ValueType> *Au, matrix::Dense<ValueType> *v)
{
    for (size_type i = 0; i < v->get_num_rows(); ++i) {
        for (size_type j = 0; j < v->get_num_cols(); ++j) {
            v->at(i, j) =
                Au_new->at(i, j) +
                beta->at(j) * (Au->at(i, j) + beta->at(j) * v->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_6_KERNEL);

template <typename ValueType>
void step_7(std::shared_ptr<const DefaultExecutor> exec,
            const matrix::Dense<ValueType> *Au_new,
            const matrix::Dense<ValueType> *u_mp1, matrix::Dense<ValueType> *Au,
            matrix::Dense<ValueType> *u_m)
{
    for (size_type i = 0; i < u_m->get_num_rows(); ++i) {
        for (size_type j = 0; j < u_m->get_num_cols(); ++j) {
            Au->at(i, j) = Au_new->at(i, j);
            u_m->at(i, j) = u_mp1->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_TFQMR_STEP_7_KERNEL);

}  // namespace tfqmr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
