// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/pipe_cg_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The PIPE_CG solver namespace.
 *
 * @ingroup pipe_cg
 */
namespace pipe_cg {


template <typename ValueType>
void initialize_1(std::shared_ptr<const ReferenceExecutor> exec,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* r,
                  matrix::Dense<ValueType>* prev_rho,
                  array<stopping_status>* stop_status)
{
    for (size_type j = 0; j < b->get_size()[1]; ++j) {
        prev_rho->at(j) = one<ValueType>();
        stop_status->get_data()[j].reset();
    }
    for (size_type i = 0; i < b->get_size()[0]; ++i) {
        for (size_type j = 0; j < b->get_size()[1]; ++j) {
            r->at(i, j) = b->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_1_KERNEL);

template <typename ValueType>
void initialize_2(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::Dense<ValueType>* p, matrix::Dense<ValueType>* q,
                  matrix::Dense<ValueType>* f, matrix::Dense<ValueType>* g,
                  matrix::Dense<ValueType>* beta,
                  const matrix::Dense<ValueType>* z,
                  const matrix::Dense<ValueType>* w,
                  const matrix::Dense<ValueType>* m,
                  const matrix::Dense<ValueType>* n,
                  const matrix::Dense<ValueType>* delta)
{
    for (size_type j = 0; j < p->get_size()[1]; ++j) {
        // beta = delta
        beta->at(j) = delta->at(j);
    }
    for (size_type i = 0; i < p->get_size()[0]; ++i) {
        // p = z
        // q = w
        // f = m
        // g = n
        for (size_type j = 0; j < p->get_size()[1]; ++j) {
            p->at(i, j) = z->at(i, j);
            q->at(i, j) = w->at(i, j);
            f->at(i, j) = m->at(i, j);
            g->at(i, j) = n->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_INITIALIZE_2_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType>* x, matrix::Dense<ValueType>* r,
            matrix::Dense<ValueType>* z, matrix::Dense<ValueType>* w,
            const matrix::Dense<ValueType>* p,
            const matrix::Dense<ValueType>* q,
            const matrix::Dense<ValueType>* f,
            const matrix::Dense<ValueType>* g,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* beta,
            const array<stopping_status>* stop_status)
{
    // tmp = rho / beta
    // x = x + tmp * p
    // r = r - tmp * q
    // z = z - tmp * f
    // w = w - tmp * g
    for (size_type i = 0; i < p->get_size()[0]; ++i) {
        for (size_type j = 0; j < p->get_size()[1]; ++j) {
            if (stop_status->get_const_data()[j].has_stopped()) {
                continue;
            }
            if (is_nonzero(beta->at(j))) {
                auto tmp = rho->at(j) / beta->at(j);
                x->at(i, j) += tmp * p->at(i, j);
                r->at(i, j) -= tmp * q->at(i, j);
                z->at(i, j) -= tmp * f->at(i, j);
                w->at(i, j) -= tmp * g->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec,
            matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* p,
            matrix::Dense<ValueType>* q, matrix::Dense<ValueType>* f,
            matrix::Dense<ValueType>* g, const matrix::Dense<ValueType>* z,
            const matrix::Dense<ValueType>* w,
            const matrix::Dense<ValueType>* m,
            const matrix::Dense<ValueType>* n,
            const matrix::Dense<ValueType>* prev_rho,
            const matrix::Dense<ValueType>* rho,
            const matrix::Dense<ValueType>* delta,
            const array<stopping_status>* stop_status)
{
    // tmp = rho / prev_rho
    // beta = delta - |tmp|^2 * beta
    // p = z + tmp * p
    // q = w + tmp * q
    // f = m + tmp * f
    // g = n + tmp * g
    for (size_type j = 0; j < p->get_size()[1]; ++j) {
        if (stop_status->get_const_data()[j].has_stopped()) {
            continue;
        }
        if (is_nonzero(prev_rho->at(j))) {
            auto tmp = rho->at(j) / prev_rho->at(j);
            auto abs_tmp = abs(tmp);
            beta->at(j) = delta->at(j) - abs_tmp * abs_tmp * beta->at(j);

            for (size_type i = 0; i < p->get_size()[0]; ++i) {
                p->at(i, j) = z->at(i, j) + tmp * p->at(i, j);
                q->at(i, j) = w->at(i, j) + tmp * q->at(i, j);
                f->at(i, j) = m->at(i, j) + tmp * f->at(i, j);
                g->at(i, j) = n->at(i, j) + tmp * g->at(i, j);
            }
        } else {
            beta->at(j) = delta->at(j);
            for (size_type i = 0; i < p->get_size()[0]; ++i) {
                p->at(i, j) = z->at(i, j);
                q->at(i, j) = w->at(i, j);
                f->at(i, j) = m->at(i, j);
                g->at(i, j) = n->at(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_STEP_2_KERNEL);


}  // namespace pipe_cg
}  // namespace reference
}  // namespace kernels
}  // namespace gko
