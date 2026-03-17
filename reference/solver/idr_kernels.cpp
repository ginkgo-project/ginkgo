// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
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
                            matrix::view::dense<const ValueType> m,
                            matrix::view::dense<const ValueType> f,
                            matrix::view::dense<ValueType> c,
                            const array<stopping_status>& stop_status)
{
    for (size_type i = 0; i < f.size[1]; i++) {
        if (stop_status.get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type row = 0; row < m.size[0]; row++) {
            auto temp = f(row, i);
            for (size_type col = 0; col < row; col++) {
                temp -= m(row, col * nrhs + i) * c(col, i);
            }
            c(row, i) = temp / m(row, row * nrhs + i);
        }
    }
}


template <typename ValueType>
void update_g_and_u(const size_type nrhs, const size_type k,
                    matrix::view::dense<const ValueType> p,
                    matrix::view::dense<const ValueType> m,
                    matrix::view::dense<ValueType> g,
                    matrix::view::dense<ValueType> g_k,
                    matrix::view::dense<ValueType> u,
                    const array<stopping_status>& stop_status)
{
    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status.get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type j = 0; j < k; j++) {
            auto alpha = zero<ValueType>();
            for (size_type ind = 0; ind < p.size[1]; ind++) {
                alpha += p(j, ind) * g_k(ind, i);
            }
            alpha /= m(j, j * nrhs + i);
            for (size_type row = 0; row < g.size[0]; row++) {
                g_k(row, i) -= alpha * g(row, j * nrhs + i);
                u(row, k * nrhs + i) -= alpha * u(row, j * nrhs + i);
            }
        }

        for (size_type row = 0; row < g.size[0]; row++) {
            g(row, k * nrhs + i) = g_k(row, i);
        }
    }
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<!is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    return static_cast<ValueType>(dist(gen));
}


template <typename ValueType, typename Distribution, typename Generator>
typename std::enable_if<is_complex_s<ValueType>::value, ValueType>::type
get_rand_value(Distribution&& dist, Generator&& gen)
{
    using real_value_type = remove_complex<ValueType>;
    return ValueType(get_rand_value<real_value_type>(dist, gen),
                     get_rand_value<real_value_type>(dist, gen));
}


}  // namespace


template <typename ValueType>
void initialize(std::shared_ptr<const ReferenceExecutor> exec,
                const size_type nrhs, matrix::view::dense<ValueType> m,
                matrix::view::dense<ValueType> subspace_vectors,
                bool deterministic, array<stopping_status>& stop_status)
{
    // Initialize M
    for (size_type i = 0; i < nrhs; i++) {
        stop_status.get_data()[i].reset();
    }

    for (size_type row = 0; row < m.size[0]; row++) {
        for (size_type col = 0; col < m.size[1]; col++) {
            m(row, col) =
                (row == col / nrhs) ? one<ValueType>() : zero<ValueType>();
        }
    }

    // Initialize and Orthonormalize P
    const auto num_rows = subspace_vectors.size[0];
    const auto num_cols = subspace_vectors.size[1];
    auto dist = std::normal_distribution<>(0.0, 1.0);
    auto seed = std::random_device{}();
    auto gen = std::default_random_engine(seed);
    for (size_type row = 0; row < num_rows; row++) {
        if (!deterministic) {
            for (size_type col = 0; col < num_cols; col++) {
                subspace_vectors(row, col) =
                    get_rand_value<ValueType>(dist, gen);
            }
        }

        for (size_type i = 0; i < row; i++) {
            auto dot = zero<ValueType>();
            for (size_type j = 0; j < num_cols; j++) {
                dot += subspace_vectors(row, j) * conj(subspace_vectors(i, j));
            }
            for (size_type j = 0; j < num_cols; j++) {
                subspace_vectors(row, j) -= dot * subspace_vectors(i, j);
            }
        }

        auto norm = zero<ValueType>();
        for (size_type j = 0; j < num_cols; j++) {
            norm += squared_norm(subspace_vectors(row, j));
        }

        norm = sqrt(norm);

        for (size_type j = 0; j < num_cols; j++) {
            subspace_vectors(row, j) /= norm;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_INITIALIZE_KERNEL);


template <typename ValueType>
void step_1(std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
            const size_type k, matrix::view::dense<const ValueType> m,
            matrix::view::dense<const ValueType> f,
            matrix::view::dense<const ValueType> residual,
            matrix::view::dense<const ValueType> g,
            matrix::view::dense<ValueType> c, matrix::view::dense<ValueType> v,
            const array<stopping_status>& stop_status)
{
    // Compute c = M \ f
    solve_lower_triangular(nrhs, m, f, c, stop_status);

    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status.get_const_data()[i].has_stopped()) {
            continue;
        }
        // v = residual - c_k * g_k - ... - c_s * g_s
        for (size_type row = 0; row < v.size[0]; row++) {
            auto temp = residual(row, i);
            for (size_type j = k; j < m.size[0]; j++) {
                temp -= c(j, i) * g(row, j * nrhs + i);
            }
            v(row, i) = temp;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_1_KERNEL);


template <typename ValueType>
void step_2(std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
            const size_type k, matrix::view::dense<const ValueType> omega,
            matrix::view::dense<const ValueType> preconditioned_vector,
            matrix::view::dense<const ValueType> c,
            matrix::view::dense<ValueType> u,
            const array<stopping_status>& stop_status)
{
    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status.get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type row = 0; row < u.size[0]; row++) {
            auto temp = omega(0, i) * preconditioned_vector(row, i);
            for (size_type j = k; j < c.size[0]; j++) {
                temp += c(j, i) * u(row, j * nrhs + i);
            }
            u(row, k * nrhs + i) = temp;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_2_KERNEL);


template <typename ValueType>
void step_3(std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
            const size_type k, matrix::view::dense<const ValueType> p,
            matrix::view::dense<ValueType> g,
            matrix::view::dense<ValueType> g_k,
            matrix::view::dense<ValueType> u, matrix::view::dense<ValueType> m,
            matrix::view::dense<ValueType> f, matrix::view::dense<ValueType>,
            matrix::view::dense<ValueType> residual,
            matrix::view::dense<ValueType> x,
            const array<stopping_status>& stop_status)
{
    update_g_and_u(nrhs, k, p, m.as_const(), g, g_k, u, stop_status);

    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status.get_const_data()[i].has_stopped()) {
            continue;
        }

        for (size_type j = k; j < m.size[0]; j++) {
            auto temp = zero<ValueType>();
            for (size_type ind = 0; ind < p.size[1]; ind++) {
                temp += p(j, ind) * g(ind, k * nrhs + i);
            }
            m(j, k * nrhs + i) = temp;
        }

        auto beta = f(k, i) / m(k, k * nrhs + i);

        for (size_type row = 0; row < g.size[0]; row++) {
            residual(row, i) -= beta * g(row, k * nrhs + i);
            x(row, i) += beta * u(row, k * nrhs + i);
        }

        if (k + 1 < f.size[0]) {
            f(k, i) = zero<ValueType>();
            for (size_type j = k + 1; j < f.size[0]; j++) {
                f(j, i) -= beta * m(j, k * nrhs + i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_STEP_3_KERNEL);


template <typename ValueType>
void compute_omega(
    std::shared_ptr<const ReferenceExecutor> exec, const size_type nrhs,
    const remove_complex<ValueType> kappa,
    matrix::view::dense<const ValueType> tht,
    matrix::view::dense<const remove_complex<ValueType>> residual_norm,
    matrix::view::dense<ValueType> omega,
    const array<stopping_status>& stop_status)
{
    for (size_type i = 0; i < nrhs; i++) {
        if (stop_status.get_const_data()[i].has_stopped()) {
            continue;
        }

        auto thr = omega(0, i);
        auto normt = sqrt(real(tht(0, i)));
        omega(0, i) /= tht(0, i);
        auto absrho = abs(thr / (normt * residual_norm(0, i)));
        if (absrho < kappa) {
            omega(0, i) *= kappa / absrho;
        }
        if (normt == zero<remove_complex<ValueType>>()) {
            omega(0, i) = 0;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_COMPUTE_OMEGA_KERNEL);


}  // namespace idr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
