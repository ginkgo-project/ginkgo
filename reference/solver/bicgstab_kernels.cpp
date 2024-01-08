// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/bicgstab_kernels.hpp"


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The BiCGSTAB solver namespace.
 *
 * @ingroup bicgstab
 */
namespace bicgstab {


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
                matrix::Dense<ValueType>* rr, matrix::Dense<ValueType>* y,
                matrix::Dense<ValueType>* s, matrix::Dense<ValueType>* t,
                matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* v,
                matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* prev_rho,
                matrix::Dense<ValueType>* rho, matrix::Dense<ValueType>* alpha,
                matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* gamma,
                matrix::Dense<ValueType>* omega,
                array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        rho->at(j) = one<ValueType>();
        prev_rho->at(j) = one<ValueType>();
        alpha->at(j) = one<ValueType>();
        beta->at(j) = one<ValueType>();
        gamma->at(j) = one<ValueType>();
        omega->at(j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b->get_size()[0]; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
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
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* v,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* prev_rho,
            const matrix::Dense<ValueType>* alpha,
            const matrix::Dense<ValueType>* omega,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < p->get_size()[0]; ++i) {
        for (size_type j = 0; j < p->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(prev_rho->at(j) * omega->at(j))) {
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
            const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* s,
            const matrix::Dense<ValueType>* v,
            const matrix::Dense<ValueType>* rho,
            matrix::Dense<ValueType>* alpha,
            const matrix::Dense<ValueType>* beta,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < s->get_size()[0]; ++i) {
        for (size_type j = 0; j < s->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta->at(j))) {
                alpha->at(j) = rho->at(j) / beta->at(j);
                s->at(i, j) = r->at(i, j) - alpha->at(j) * v->at(i, j);
            } else {
                alpha->at(j) = zero<ValueType>();
                s->at(i, j) = r->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_2_KERNEL);


template <typename ValueType>
void step_3(
    std::shared_ptr<const ReferenceExecutor> exec, matrix::Dense<ValueType>* x,
    matrix::Dense<ValueType>* r, const matrix::Dense<ValueType>* s,
    const matrix::Dense<ValueType>* t, const matrix::Dense<ValueType>* y,
    const matrix::Dense<ValueType>* z, const matrix::Dense<ValueType>* alpha,
    const matrix::Dense<ValueType>* beta, const matrix::Dense<ValueType>* gamma,
    matrix::Dense<ValueType>* omega, const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(beta->at(j))) {
            omega->at(j) = gamma->at(j) / beta->at(j);
        } else {
            omega->at(j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            x->at(i, j) +=
                alpha->at(j) * y->at(i, j) + omega->at(j) * z->at(i, j);
            r->at(i, j) = s->at(i, j) - omega->at(j) * t->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_STEP_3_KERNEL);


template <typename ValueType>
void finalize(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::Dense<ValueType>* x, const matrix::Dense<ValueType>* y,
              const matrix::Dense<ValueType>* alpha,
              array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < x->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped() &&
            !stop_status->get_const_data()[j].is_finalized()) {
            for (size_type i = 0; i < x->get_size()[0]; ++i) {
                x->at(i, j) += alpha->at(j) * y->at(i, j);
                stop_status->get_data()[j].finalize();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BICGSTAB_FINALIZE_KERNEL);


}  // namespace bicgstab
}  // namespace reference
}  // namespace kernels
}  // namespace gko
