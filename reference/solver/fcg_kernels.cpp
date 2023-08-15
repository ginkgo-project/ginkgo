// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/fcg_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The FCG solver namespace.
 *
 * @ingroup fcg
 */
namespace fcg {


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* r,
                matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* p,
                matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* t,
                matrix::Dense<ValueType>* prev_rho,
                matrix::Dense<ValueType>* rho, matrix::Dense<ValueType>* rho_t,
                array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        rho->at(j) = zero<ValueType>();
        prev_rho->at(j) = one<ValueType>();
        rho_t->at(j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b->get_size()[0]; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
            t->at(i, j) = r->at(i, j) = b->at(i, j);
            z->at(i, j) = p->at(i, j) = q->at(i, j) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType>* p, const matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* rho_t,
            const matrix::Dense<ValueType>* prev_rho,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < p->get_size()[0]; ++i) {
        for (size_type j = 0; j < p->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_zero(prev_rho->at(j))) {
                p->at(i, j) = z->at(i, j);
            } else {
                auto tmp = rho_t->at(j) / prev_rho->at(j);
                p->at(i, j) = z->at(i, j) + tmp * p->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
            matrix::Dense<ValueType>* t, const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* q,
            const matrix::Dense<ValueType>* beta,
            const matrix::Dense<ValueType>* rho,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < x->get_size()[0]; ++i) {
        for (size_type j = 0; j < x->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta->at(j))) {
                auto tmp = rho->at(j) / beta->at(j);
                auto prev_r = r->at(i, j);
                x->at(i, j) += tmp * p->at(i, j);
                r->at(i, j) -= tmp * q->at(i, j);
                t->at(i, j) = r->at(i, j) - prev_r;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FCG_STEP_2_KERNEL);


}  // namespace fcg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
