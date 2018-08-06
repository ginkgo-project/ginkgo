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


#include <iostream>


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
                  matrix::Dense<ValueType> *b_norm, Array<size_type> *iter_nums,
                  Array<stopping_status> *stop_status)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        stop_status->get_data()[j].reset();
        iter_nums->get_data()[j] = 0;
        b_norm->at(0, j) = zero<ValueType>();
        for (size_type i = 0; i < b->get_size()[0]; ++i) {
            b_norm->at(0, j) += b->at(i, j) * b->at(i, j);
        }
        b_norm->at(0, j) = sqrt(b_norm->at(0, j));
    }
    for (size_type i = 0; i < b->get_size()[0]; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
            r->at(i, j) = b->at(i, j);
            if (i == 0) {
                e1->at(i, j) = one<ValueType>();
            } else {
                e1->at(i, j) = zero<ValueType>();
            }
        }
    }
    for (size_type i = 0; i < solver::default_max_num_iterations; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
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
    for (size_type i = 0; i < r->get_size()[1]; ++i) {
        r_norm->at(0, i) = 0;
        for (size_type j = 0; j < r->get_size()[0]; ++j) {
            r_norm->at(0, i) += r->at(j, i) * r->at(j, i);
        }
        r_norm->at(0, i) = sqrt(r_norm->at(0, i));
    }
    for (size_type i = 0; i < r->get_size()[0]; ++i) {
        for (size_type j = 0; j < r->get_size()[1]; ++j) {
            range_Q(i, j) = r->at(i, j) / r_norm->at(0, j);
        }
    }
    for (size_type i = 0; i < solver::default_max_num_iterations + 1; ++i) {
        for (size_type j = 0; j < r->get_size()[1]; ++j) {
            if (i == 0) {
                beta->at(i, j) = r_norm->at(0, j);
            } else {
                beta->at(i, j) = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_INITIALIZE_2_KERNEL);


template <typename ValueType, typename AccessorType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType> *q, matrix::Dense<ValueType> *sn,
            matrix::Dense<ValueType> *cs, matrix::Dense<ValueType> *beta,
            AccessorType range_Q, AccessorType range_H_k,
            matrix::Dense<ValueType> *r_norm,
            const matrix::Dense<ValueType> *b_norm, const size_type iter_id,
            const Array<stopping_status> *stop_status)
{
    for (size_type k = 0; k < iter_id + 1; ++k) {
        for (size_type i = 0; i < q->get_size()[1]; ++i) {
            if (stop_status->get_const_data()[i].has_stopped()) {
                continue;
            }
            range_H_k(k, i) = 0;
            for (size_type j = 0; j < q->get_size()[0]; ++j) {
                range_H_k(k, i) +=
                    q->at(j, i) * range_Q(j, q->get_size()[1] * k + i);
            }
            for (size_type j = 0; j < q->get_size()[0]; ++j) {
                q->at(j, i) -=
                    range_H_k(k, i) * range_Q(j, q->get_size()[1] * k + i);
            }
        }
    }

    for (size_type i = 0; i < q->get_size()[1]; ++i) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }
        range_H_k(iter_id + 1, i) = 0;
        for (size_type j = 0; j < q->get_size()[0]; ++j) {
            range_H_k(iter_id + 1, i) += q->at(j, i) * q->at(j, i);
        }
        range_H_k(iter_id + 1, i) = sqrt(range_H_k(iter_id + 1, i));
    }

    for (size_type i = 0; i < q->get_size()[1]; ++i) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }
        for (size_type j = 0; j < q->get_size()[0]; ++j) {
            q->at(j, i) /= range_H_k(iter_id + 1, i);
            range_Q(j, q->get_size()[1] * (iter_id + 1) + i) = q->at(j, i);
        }
    }
    // End of arnoldi

    for (size_type i = 0; i < q->get_size()[1]; ++i) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }
        // Start apply givens rotation
        for (size_type j = 0; j < iter_id; ++j) {
            auto temp = cs->at(j, i) * range_H_k(j, i) +
                        sn->at(j, i) * range_H_k(j + 1, i);
            range_H_k(j + 1, i) = -sn->at(j, i) * range_H_k(j, i) +
                                  cs->at(j, i) * range_H_k(j + 1, i);
            range_H_k(j, i) = temp;
        }
        if (range_H_k(iter_id, i) == zero<ValueType>()) {
            cs->at(iter_id, i) = zero<ValueType>();
            sn->at(iter_id, i) = one<ValueType>();
        } else {
            auto t =
                sqrt(range_H_k(iter_id, i) * range_H_k(iter_id, i) +
                     range_H_k(iter_id + 1, i) * range_H_k(iter_id + 1, i));
            cs->at(iter_id, i) = abs(range_H_k(iter_id, i)) / t;
            sn->at(iter_id, i) = cs->at(iter_id, i) *
                                 range_H_k(iter_id + 1, i) /
                                 range_H_k(iter_id, i);
        }
        range_H_k(iter_id, i) = cs->at(iter_id, i) * range_H_k(iter_id, i) +
                                sn->at(iter_id, i) * range_H_k(iter_id + 1, i);
        range_H_k(iter_id + 1, i) = zero<ValueType>();
        // End apply givens rotation

        beta->at(iter_id + 1, i) = -sn->at(iter_id, i) * beta->at(iter_id, i);
        beta->at(iter_id, i) = cs->at(iter_id, i) * beta->at(iter_id, i);
        r_norm->at(0, i) = abs(beta->at(iter_id + 1, i)) / b_norm->at(0, i);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_STEP_1_KERNEL);


template <typename ValueType, typename AccessorType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType> *beta, AccessorType range_H,
            const Array<size_type> *iter_nums, matrix::Dense<ValueType> *y,
            AccessorType range_Q, matrix::Dense<ValueType> *x)
{
    // Solve upper triangular.
    for (size_type k = 0; k < beta->get_size()[1]; ++k) {
        for (int i = iter_nums->get_const_data()[k] - 1; i >= 0; --i) {
            auto temp = beta->at(i, k);
            for (size_type j = i + 1; j < iter_nums->get_const_data()[k]; ++j) {
                temp -= range_H(i, j * beta->get_size()[1] + k) * y->at(j, k);
            }
            y->at(i, k) = temp / range_H(i, i * beta->get_size()[1] + k);
        }
    }

    // Solve x
    for (size_type k = 0; k < x->get_size()[1]; ++k) {
        for (size_type i = 0; i < x->get_size()[0]; ++i) {
            for (size_type j = 0; j < iter_nums->get_const_data()[k]; ++j) {
                x->at(i, k) +=
                    range_Q(i, j * x->get_size()[1] + k) * y->at(j, k);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_ACCESSOR_TYPE(
    GKO_DECLARE_GMRES_STEP_2_KERNEL);


}  // namespace gmres
}  // namespace reference
}  // namespace kernels
}  // namespace gko
