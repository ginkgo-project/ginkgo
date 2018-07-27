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

#include "core/solver/gmres_kernels.hpp"


#include "core/base/array.hpp"
#include "core/base/exception_helpers.hpp"
#include "core/base/math.hpp"
#include "core/base/types.hpp"
#include "core/solver/gmres.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace gmres {


template <typename ValueType>
void initialize_1(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *r, matrix::Dense<ValueType> *e1,
                  matrix::Dense<ValueType> *sn, matrix::Dense<ValueType> *cs,
                  Array<stopping_status> *stop_status)
{
    for (size_type j = 0; j < b->get_size().num_cols; ++j) {
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b->get_size().num_rows; ++i) {
        for (size_type j = 0; j < b->get_size().num_cols; ++j) {
            r->at(i, j) = b->at(i, j);
            if (i == 0) {
                e1->at(i, j) = one<ValueType>();
            } else {
                e1->at(i, j) = zero<ValueType>();
            }
        }
    }
    for (size_type i = 0; i < solver::default_max_num_iterations; ++i) {
        for (size_type j = 0; j < b->get_size().num_cols; ++j) {
            sn->at(i, j) = zero<ValueType>();
            cs->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_INITIALIZE_1_KERNEL);


template <typename ValueType, typename AccessorType>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType> *r,
                  matrix::Dense<ValueType> *r_norm,
                  matrix::Dense<ValueType> *beta, AccessorType range_Q)
{
    for (int i = 0; i < r->get_size().num_cols; ++i) {
        r_norm->at(0, i) = 0;
        for (int j = 0; j < r->get_size().num_rows; ++j) {
            r_norm->at(0, i) += r->at(i, j) * r->at(i, j);
        }
        r_norm->at(0, i) = sqrt(r_norm->at(0, i));
    }
    for (size_type i = 0; i < r->get_size().num_rows; ++i) {
        for (size_type j = 0; j < r->get_size().num_cols; ++j) {
            if (i == 0) {
                beta->at(i, j) = r_norm->at(0, j);
            } else {
                beta->at(i, j) = zero<ValueType>();
            }
            range_Q(i, j) = r->at(i, j) / r_norm->at(0, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *p, const matrix::Dense<ValueType> *z,
            const matrix::Dense<ValueType> *rho,
            const matrix::Dense<ValueType> *prev_rho,
            const Array<stopping_status> *stop_status)
{
    for (size_type i = 0; i < p->get_size().num_rows; ++i) {
        for (size_type j = 0; j < p->get_size().num_cols; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (prev_rho->at(j) == zero<ValueType>()) {
                p->at(i, j) = z->at(i, j);
            } else {
                auto tmp = rho->at(j) / prev_rho->at(j);
                p->at(i, j) = z->at(i, j) + tmp * p->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *x, matrix::Dense<ValueType> *r,
            const matrix::Dense<ValueType> *p,
            const matrix::Dense<ValueType> *q,
            const matrix::Dense<ValueType> *beta,
            const matrix::Dense<ValueType> *rho,
            const Array<stopping_status> *stop_status)
{
    for (size_type i = 0; i < x->get_size().num_rows; ++i) {
        for (size_type j = 0; j < x->get_size().num_cols; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (beta->at(j) != zero<ValueType>()) {
                auto tmp = rho->at(j) / beta->at(j);
                x->at(i, j) += tmp * p->at(i, j);
                r->at(i, j) -= tmp * q->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
