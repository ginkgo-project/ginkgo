// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/cgs_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The CGS solver namespace.
 *
 * @ingroup cgs
 */
namespace cgs {


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
                matrix::Dense<ValueType>* r_tld, matrix::Dense<ValueType>* p,
                matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* u,
                matrix::Dense<ValueType>* u_hat,
                matrix::Dense<ValueType>* v_hat, matrix::Dense<ValueType>* t,
                matrix::Dense<ValueType>* alpha, matrix::Dense<ValueType>* beta,
                matrix::Dense<ValueType>* gamma,
                matrix::Dense<ValueType>* rho_prev,
                matrix::Dense<ValueType>* rho,
                array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        rho->at(j) = zero<ValueType>();
        rho_prev->at(j) = one<ValueType>();
        alpha->at(j) = one<ValueType>();
        beta->at(j) = one<ValueType>();
        gamma->at(j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b->get_size()[0]; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
            r->at(i, j) = b->at(i, j);
            r_tld->at(i, j) = b->at(i, j);
            u->at(i, j) = u_hat->at(i, j) = p->at(i, j) = q->at(i, j) =
                v_hat->at(i, j) = t->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType>* r, matrix::Dense<ValueType>* u,
            matrix::Dense<ValueType>* p, const matrix::Dense<ValueType>* q,
            matrix::Dense<ValueType>* beta, const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* rho_prev,
            const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < p->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(rho_prev->at(j))) {
            beta->at(j) = rho->at(j) / rho_prev->at(j);
        }
    }
    for (size_type i = 0; i < p->get_size()[0]; ++i) {
        for (size_type j = 0; j < p->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            u->at(i, j) = r->at(i, j) + beta->at(j) * q->at(i, j);
            p->at(i, j) =
                u->at(i, j) +
                beta->at(j) * (q->at(i, j) + beta->at(j) * p->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType>* u,
            const matrix::Dense<ValueType>* v_hat, matrix::Dense<ValueType>* q,
            matrix::Dense<ValueType>* t, matrix::Dense<ValueType>* alpha,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* gamma,
            const array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < u->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(gamma->at(j))) {
            alpha->at(j) = rho->at(j) / gamma->at(j);
        }
    }
    for (size_type i = 0; i < u->get_size()[0]; ++i) {
        for (size_type j = 0; j < u->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            q->at(i, j) = u->at(i, j) - alpha->at(j) * v_hat->at(i, j);
            t->at(i, j) = u->at(i, j) + q->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Dense<ValueType>* t,
            const matrix::Dense<ValueType>* u_hat, matrix::Dense<ValueType>* r,
            matrix::Dense<ValueType>* x, const matrix::Dense<ValueType>* alpha,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            x->at(i, j) += alpha->at(j) * u_hat->at(i, j);
            r->at(i, j) -= alpha->at(j) * t->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CGS_STEP_3_KERNEL);


}  // namespace cgs
}  // namespace reference
}  // namespace kernels
}  // namespace gko
