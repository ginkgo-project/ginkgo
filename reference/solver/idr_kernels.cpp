// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/idr_kernels.hpp"


#include <algorithm>
#include <ctime>
#include <random>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The IDR solver namespace.
 *
 * @ingroup idr
 */
namespace idr {


namespace {


template <typename ValueType>
void solve_lower_triangular(const size_type nrhs,
                            const matrix::Dense<ValueType>* m,
                            const matrix::Dense<ValueType>* f,
                            matrix::Dense<ValueType>* c,
                            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < f->get_size()[1]; i++) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type row = 0; row < m->get_size()[0]; row++) {
            auto temp = f->at(row, i);
            for (size_type col = 0; col < row; col++) {
                temp -= m->at(row, col * nrhs + i) * c->at(col, i);
            }
            c->at(row, i) = temp / m->at(row, row * nrhs + i);
        }
    }
}


template <typename ValueType>
void update_g_and_u(const size_type nrhs, const size_type k,
                    const matrix::Dense<ValueType>* p,
                    const matrix::Dense<ValueType>* m,
                    matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
                    matrix::Dense<ValueType>* u,
                    const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type j = 0; j < k; j++) {
            auto alpha = zero<ValueType>();
            for (size_type ind = 0; ind < p->get_size()[1]; ind++) {
                alpha += p->at(j, ind) * g_k->at(ind, i);
            }
            alpha /= m->at(j, j * nrhs + i);
            for (size_type row = 0; row < g->get_size()[0]; row++) {
                g_k->at(row, i) -= alpha * g->at(row, j * nrhs + i);
                u->at(row, k * nrhs + i) -= alpha * u->at(row, j * nrhs + i);
            }
        }

        for (size_type row = 0; row < g->get_size()[0]; row++) {
            g->at(row, k * nrhs + i) = g_k->at(row, i);
        }
    }
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return dist(gen);
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return ValueType(dist(gen), dist(gen));
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const size_type nrhs, matrix::Dense<ValueType>* m,
                matrix::Dense<ValueType>* subspace_vectors, bool deterministic,
                array<stopping_status>* stop_status)
{
    // Initialize M
    for (size_type i = 0; i < nrhs; i++) {
        stop_status->get_data()[i].reset();
    }

    for (size_type row = 0; row < m->get_size()[0]; row++) {
        for (size_type col = 0; col < m->get_size()[1]; col++) {
            m->at(row, col) =
                (row == col / nrhs) ? one<ValueType>() : zero<ValueType>();
        }
    }

    // Initialize and Orthonormalize P
    const auto num_rows = subspace_vectors->get_size()[0];
    const auto num_cols = subspace_vectors->get_size()[1];
    auto dist = std::normal_distribution<remove_complex<ValueType>>(0.0, 1.0);
    auto seed = std::random_device{}();
    auto gen = std::default_random_engine(seed);
    for (size_type row = 0; row < num_rows; row++) {
        if (!deterministic) {
            for (size_type col = 0; col < num_cols; col++) {
                subspace_vectors->at(row, col) =
                    get_rand_value<ValueType>(dist, gen);
            }
        }

        for (size_type i = 0; i < row; i++) {
            auto dot = zero<ValueType>();
            for (size_type j = 0; j < num_cols; j++) {
                dot += subspace_vectors->at(row, j) *
                       conj(subspace_vectors->at(i, j));
            }
            for (size_type j = 0; j < num_cols; j++) {
                subspace_vectors->at(row, j) -=
                    dot * subspace_vectors->at(i, j);
            }
        }

        auto norm = zero<ValueType>();
        for (size_type j = 0; j < num_cols; j++) {
            norm += squared_norm(subspace_vectors->at(row, j));
        }

        norm = sqrt(norm);

        for (size_type j = 0; j < num_cols; j++) {
            subspace_vectors->at(row, j) /= norm;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* m,
            const matrix::Dense<ValueType>* f,
            const matrix::Dense<ValueType>* residual,
            const matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* c,
            matrix::Dense<ValueType>* v,
            const array<stopping_status>* stop_status)
{
    // Compute c = M \ f
    solve_lower_triangular(nrhs, m, f, c, stop_status);

    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }
        // v = residual - c_k * g_k - ... - c_s * g_s
        for (size_type row = 0; row < v->get_size()[0]; row++) {
            auto temp = residual->at(row, i);
            for (size_type j = k; j < m->get_size()[0]; j++) {
                temp -= c->at(j, i) * g->at(row, j * nrhs + i);
            }
            v->at(row, i) = temp;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* omega,
            const matrix::Dense<ValueType>* preconditioned_vector,
            const matrix::Dense<ValueType>* c, matrix::Dense<ValueType>* u,
            const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type row = 0; row < u->get_size()[0]; row++) {
            auto temp = omega->at(0, i) * preconditioned_vector->at(row, i);
            for (size_type j = k; j < c->get_size()[0]; j++) {
                temp += c->at(j, i) * u->at(row, j * nrhs + i);
            }
            u->at(row, k * nrhs + i) = temp;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
            const size_type k, const matrix::Dense<ValueType>* p,
            matrix::Dense<ValueType>* g, matrix::Dense<ValueType>* g_k,
            matrix::Dense<ValueType>* u, matrix::Dense<ValueType>* m,
            matrix::Dense<ValueType>* f, matrix::Dense<ValueType>*,
            matrix::Dense<ValueType>* residual, matrix::Dense<ValueType>* x,
            const array<stopping_status>* stop_status)
{
    update_g_and_u(nrhs, k, p, m, g, g_k, u, stop_status);

    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type j = k; j < m->get_size()[0]; j++) {
            auto temp = zero<ValueType>();
            for (size_type ind = 0; ind < p->get_size()[1]; ind++) {
                temp += p->at(j, ind) * g->at(ind, k * nrhs + i);
            }
            m->at(j, k * nrhs + i) = temp;
        }

        auto beta = f->at(k, i) / m->at(k, k * nrhs + i);

        for (size_type row = 0; row < g->get_size()[0]; row++) {
            residual->at(row, i) -= beta * g->at(row, k * nrhs + i);
            x->at(row, i) += beta * u->at(row, k * nrhs + i);
        }

        if (k + 1 < f->get_size()[0]) {
            f->at(k, i) = zero<ValueType>();
            for (size_type j = k + 1; j < f->get_size()[0]; j++) {
                f->at(j, i) -= beta * m->at(j, k * nrhs + i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa, const matrix::Dense<ValueType>* tht,
    const matrix::Dense<remove_complex<ValueType>>* residual_norm,
    matrix::Dense<ValueType>* omega, const array<stopping_status>* stop_status)
{
    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status->get_const_data()[i].has_stopped()) {
            continue;
        }

        auto thr = omega->at(0, i);
        auto normt = sqrt(real(tht->at(0, i)));
        omega->at(0, i) /= tht->at(0, i);
        auto absrho = abs(thr / (normt * residual_norm->at(0, i)));

        if (absrho < kappa) {
            omega->at(0, i) *= kappa / absrho;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
