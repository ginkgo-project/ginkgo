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

#include "core/solver/bicgstab_kernels.hpp"

#include "core/base/array.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"

#include <algorithm>


namespace gko {
namespace kernels {
namespace reference {
namespace bicgstab {


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *r,
                matrix::Dense<ValueType> *rr, matrix::Dense<ValueType> *y,
                matrix::Dense<ValueType> *s, matrix::Dense<ValueType> *t,
                matrix::Dense<ValueType> *z, matrix::Dense<ValueType> *v,
                matrix::Dense<ValueType> *p, matrix::Dense<ValueType> *prev_rho,
                matrix::Dense<ValueType> *rho, matrix::Dense<ValueType> *alpha,
                matrix::Dense<ValueType> *beta, matrix::Dense<ValueType> *gamma,
                matrix::Dense<ValueType> *omega, Array<bool> *converged)
{
    for (size_type j = 0; j < b->get_size().num_cols; ++j) {
        rho->at(j) = one<ValueType>();
        prev_rho->at(j) = one<ValueType>();
        alpha->at(j) = one<ValueType>();
        beta->at(j) = one<ValueType>();
        gamma->at(j) = one<ValueType>();
        omega->at(j) = one<ValueType>();
        converged->get_data()[j] = false;
    }
    for (size_type i = 0; i < b->get_size().num_rows; ++i) {
        for (size_type j = 0; j < b->get_size().num_cols; ++j) {
            r->at(i, j) = b->at(i, j);
            rr->at(i, j) = zero<ValueType>();
            z->at(i, j) = zero<ValueType>();
            v->at(i, j) = zero<ValueType>();
            s->at(i, j) = zero<ValueType>();
            t->at(i, j) = zero<ValueType>();
            y->at(i, j) = zero<ValueType>();
            p->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_INITIALIZE_KERNEL);


template <typename ValueType>
void test_convergence(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Dense<ValueType> *tau,
                      const matrix::Dense<ValueType> *orig_tau,
                      remove_complex<ValueType> rel_residual_goal,
                      Array<bool> *converged, bool *all_converged)
{
    *all_converged = true;
    for (size_type i = 0; i < tau->get_size().num_cols; ++i) {
        if (abs(tau->at(i)) < rel_residual_goal * abs(orig_tau->at(i))) {
            converged->get_data()[i] = true;
        }
    }
    for (size_type i = 0; i < converged->get_num_elems(); ++i) {
        if (!converged->get_const_data()[i]) {
            *all_converged = false;
            break;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BICGSTAB_TEST_CONVERGENCE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *omega, const Array<bool> &converged)

{
    for (size_type i = 0; i < p->get_size().num_rows; ++i) {
        for (size_type j = 0; j < p->get_size().num_cols; ++j) {
            if (converged.get_const_data()[j]) {
                continue;
            }
            if (prev_rho->at(j) * omega->at(j) != zero<ValueType>()) {
                const auto tmp =
                    rho->at(j) / prev_rho->at(j) * alpha->at(j) / omega->at(j);
                p->at(i, j) = r->at(i, j) +
                              tmp * (p->at(i, j) - omega->at(j) * v->at(i, j));
            } else {
                p->at(i, j) = r->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *v,
            const matrix::Dense<ValueType> *rho,
            matrix::Dense<ValueType> *alpha,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *y, matrix::Dense<ValueType> *x,
            const Array<bool> &converged)
{
    for (size_type i = 0; i < s->get_size().num_rows; ++i) {
        for (size_type j = 0; j < s->get_size().num_cols; ++j) {
            if (converged.get_const_data()[j]) {
                continue;
            }
            if (beta->at(j) != zero<ValueType>()) {
                alpha->at(j) = rho->at(j) / beta->at(j);
                s->at(i, j) = r->at(i, j) - alpha->at(j) * v->at(i, j);
            } else {
                alpha->at(j) = zero<ValueType>();
                s->at(i, j) = r->at(i, j);
            }
            x->at(i, j) += alpha->at(j) * y->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *s,
            const matrix::Dense<ValueType> *t,
            const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *gamma,
            matrix::Dense<ValueType> *omega, const Array<bool> &converged)
{
    for (size_type j = 0; j < x->get_size().num_cols; ++j) {
        if (converged.get_const_data()[j]) {
            continue;
        }
        if (beta->at(j) != zero<ValueType>()) {
            omega->at(j) = gamma->at(j) / beta->at(j);
        } else {
            omega->at(j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < x->get_size().num_rows; ++i) {
        for (size_type j = 0; j < x->get_size().num_cols; ++j) {
            if (converged.get_const_data()[j]) {
                continue;
            }
            x->at(i, j) += omega->at(j) * z->at(i, j);
            r->at(i, j) = s->at(i, j) - omega->at(j) * t->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);

}  // namespace bicgstab
}  // namespace reference
}  // namespace kernels
}  // namespace gko
