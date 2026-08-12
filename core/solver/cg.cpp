// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/cg.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/nullspace_removable.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>

#include "core/config/solver_config.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/cg_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace cg {
namespace {


GKO_REGISTER_OPERATION(initialize, cg::initialize);
GKO_REGISTER_OPERATION(step_1, cg::step_1);
GKO_REGISTER_OPERATION(step_2, cg::step_2);


}  // anonymous namespace
}  // namespace cg


template <typename ValueType>
typename Cg<ValueType>::parameters_type Cg<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::Cg<ValueType>::build();
    common_solver_parse(params, config, context, td_for_child);
    return params;
}


template <typename ValueType>
std::unique_ptr<LinOp> Cg<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Cg<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
void Cg<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(z, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(q, dense_b);

    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0
    exec->run(cg::make_initialize(
        gko::detail::get_local(dense_b), gko::detail::get_local(r),
        gko::detail::get_local(z), gko::detail::get_local(p),
        gko::detail::get_local(q), prev_rho, rho, &stop_status));

    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    // If the system matrix is singular with a known nullspace (e.g. the
    // constant vector for a pure-Neumann problem), project it out following
    // PETSc's MatSetNullSpace convention: the initial residual (which carries
    // the right-hand side) is projected so the system is consistent, and each
    // search direction is projected so the iterate stays orthogonal to the
    // nullspace.
    auto nullspace_op = dynamic_cast<const NullspaceRemovable*>(
        this->get_system_matrix().get());
    const bool project_nullspace =
        nullspace_op && nullspace_op->has_nullspace();
    if (project_nullspace) {
        nullspace_op->remove_nullspace(r);
    }
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);

    int iter = -1;
    /* Memory movement summary:
     * 18n * values + matrix/preconditioner storage
     * 1x SpMV:           2n * values + storage
     * 1x Preconditioner: 2n * values + storage
     * 2x dot             4n
     * 1x step 1 (axpy)   3n
     * 1x step 2 (axpys)  6n
     * 1x norm2 residual   n
     */
    while (true) {
        // z = preconditioner * r
        this->get_preconditioner()->apply(r, z);
        // rho = dot(r, z)
        r->compute_conj_dot(z, rho, reduction_tmp);

        ++iter;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, iter, r, nullptr, rho, &stop_status,
            all_stopped);
        if (all_stopped) {
            break;
        }

        // tmp = rho / prev_rho
        // p = z + tmp * p
        exec->run(cg::make_step_1(gko::detail::get_local(p),
                                  gko::detail::get_local(z), rho, prev_rho,
                                  &stop_status));
        // keep the search direction orthogonal to the nullspace
        if (project_nullspace) {
            nullspace_op->remove_nullspace(p);
        }
        // q = A * p
        this->get_system_matrix()->apply(p, q);
        // beta = dot(p, q)
        p->compute_conj_dot(q, beta, reduction_tmp);
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(cg::make_step_2(
            gko::detail::get_local(dense_x), gko::detail::get_local(r),
            gko::detail::get_local(p), gko::detail::get_local(q), beta, rho,
            &stop_status));
        swap(prev_rho, rho);
    }
    // Remove any nullspace component that accumulated in the solution so the
    // returned x is the minimum-norm solution orthogonal to the nullspace.
    if (project_nullspace) {
        nullspace_op->remove_nullspace(dense_x);
    }
}


template <typename ValueType>
void Cg<ValueType>::condest(const LinOp* b, LinOp* eigs) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_eigs) {
            this->condest_impl(dense_b, dense_eigs);
        },
        b, eigs);
}


template <typename ValueType>
template <typename VectorType>
void Cg<ValueType>::condest_impl(const VectorType* dense_b,
                                 VectorType* dense_eigs) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;
    using real_type = remove_complex<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    this->setup_workspace();

    auto x = share(clone(dense_b));
    auto dense_x = x.get();
    dense_x->fill(zero<ValueType>());
    GKO_SOLVER_VECTOR(r, dense_b);
    GKO_SOLVER_VECTOR(z, dense_b);
    GKO_SOLVER_VECTOR(p, dense_b);
    GKO_SOLVER_VECTOR(q, dense_b);

    GKO_SOLVER_SCALAR(beta, dense_b);
    GKO_SOLVER_SCALAR(prev_rho, dense_b);
    GKO_SOLVER_SCALAR(rho, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    std::vector<ValueType> est_alpha;
    est_alpha.emplace_back(one<ValueType>());
    std::vector<ValueType> est_beta;
    auto host_rho = clone(exec->get_master(), rho);
    auto host_beta = clone(exec->get_master(), beta);

    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0
    exec->run(cg::make_initialize(
        gko::detail::get_local(dense_b), gko::detail::get_local(r),
        gko::detail::get_local(z), gko::detail::get_local(p),
        gko::detail::get_local(q), prev_rho, rho, &stop_status));

    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, r);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x, r);

    int iter = -1;
    /* Memory movement summary:
     * 18n * values + matrix/preconditioner storage
     * 1x SpMV:           2n * values + storage
     * 1x Preconditioner: 2n * values + storage
     * 2x dot             4n
     * 1x step 1 (axpy)   3n
     * 1x step 2 (axpys)  6n
     * 1x norm2 residual   n
     */
    while (true) {
        // z = preconditioner * r
        this->get_preconditioner()->apply(r, z);
        // rho = dot(r, z)
        r->compute_conj_dot(z, rho, reduction_tmp);

        ++iter;
        bool all_stopped =
            stop_criterion->update()
                .num_iterations(iter)
                .residual(r)
                .implicit_sq_residual_norm(rho)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, iter, r, nullptr, rho, &stop_status,
            all_stopped);
        if (all_stopped) {
            auto n = est_alpha.size();
            // The CG step lengths (est_alpha) and update coefficients
            // (est_beta) define the Lanczos/Jacobi tridiagonal matrix T whose
            // eigenvalues (the Ritz values) approximate the spectrum of the
            // (preconditioned) operator:
            //   T_ii      = 1/alpha_i + beta_{i-1}/alpha_{i-1}
            //   T_{i,i-1} = sqrt(beta_i)/alpha_i
            // We skip the very first Ritz index and the last few iterations,
            // whose Ritz values are the least converged.
            constexpr size_type num_skipped = 4;
            if (n <= num_skipped) {
                dense_eigs->fill(one<ValueType>());
            } else {
                const auto m = n - num_skipped;  // size of T
                std::vector<real_type> t_diag(m);
                std::vector<real_type> t_off(m > 0 ? m - 1 : 0);
                for (size_type i = 0; i < m; i++) {
                    t_diag[i] = real(one<ValueType>() / est_alpha[i + 2] +
                                     est_beta[i + 1] / est_alpha[i + 1]);
                }
                for (size_type i = 0; i + 1 < m; i++) {
                    t_off[i] = real(sqrt(est_beta[i + 2]) / est_alpha[i + 2]);
                }
                constexpr auto tiny = std::numeric_limits<real_type>::min();
                // Number of eigenvalues of T strictly below sigma, via the
                // Sturm sequence (sign changes of the LDL^T pivots of
                // T - sigma*I). This is the standard symmetric-tridiagonal
                // eigenvalue count used for bisection.
                auto count_less = [&](real_type sigma) {
                    int count = 0;
                    real_type q = t_diag[0] - sigma;
                    count += q < zero<real_type>();
                    for (size_type i = 1; i < m; i++) {
                        if (q == zero<real_type>()) {
                            q = tiny;
                        }
                        q = (t_diag[i] - sigma) - t_off[i - 1] * t_off[i - 1] / q;
                        count += q < zero<real_type>();
                    }
                    return count;
                };
                // Gershgorin disks bracket the whole spectrum of T.
                real_type lower = t_diag[0];
                real_type upper = t_diag[0];
                for (size_type i = 0; i < m; i++) {
                    real_type radius = zero<real_type>();
                    if (i > 0) {
                        radius += abs(t_off[i - 1]);
                    }
                    if (i + 1 < m) {
                        radius += abs(t_off[i]);
                    }
                    lower = min(lower, t_diag[i] - radius);
                    upper = max(upper, t_diag[i] + radius);
                }
                // Bisect for the k-th smallest eigenvalue of T.
                auto find_eig = [&](int k) {
                    real_type lo = lower;
                    real_type hi = upper;
                    for (int it = 0;
                         it < 100 &&
                         hi - lo > std::numeric_limits<real_type>::epsilon() *
                                       (abs(lo) + abs(hi) + tiny);
                         it++) {
                        real_type mid = lo + (hi - lo) / 2;
                        if (count_less(mid) >= k) {
                            hi = mid;
                        } else {
                            lo = mid;
                        }
                    }
                    return lo + (hi - lo) / 2;
                };
                gko::detail::get_local(dense_eigs)->at(0, 0) = find_eig(1);
                gko::detail::get_local(dense_eigs)->at(1, 0) =
                    find_eig(static_cast<int>(m));
                break;
            }
        }

        host_rho->copy_from(rho);
        host_beta->copy_from(prev_rho);
        est_beta.emplace_back(host_rho->at(0, 0) / host_beta->at(0, 0));
        // tmp = rho / prev_rho
        // p = z + tmp * p
        exec->run(cg::make_step_1(gko::detail::get_local(p),
                                  gko::detail::get_local(z), rho, prev_rho,
                                  &stop_status));
        // q = A * p
        this->get_system_matrix()->apply(p, q);
        // beta = dot(p, q)
        p->compute_conj_dot(q, beta, reduction_tmp);
        host_beta->copy_from(beta);
        est_alpha.emplace_back(host_rho->at(0, 0) / host_beta->at(0, 0));
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(cg::make_step_2(
            gko::detail::get_local(dense_x), gko::detail::get_local(r),
            gko::detail::get_local(p), gko::detail::get_local(q), beta, rho,
            &stop_status));
        swap(prev_rho, rho);
    }
}

template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Cg<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<Cg<ValueType>>::num_vectors(const Solver&)
{
    return 9;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Cg<ValueType>>::op_names(
    const Solver&)
{
    return {
        "r", "z", "p", "q", "beta", "prev_rho", "rho", "one", "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Cg<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Cg<ValueType>>::scalars(const Solver&)
{
    return {beta, prev_rho, rho};
}


template <typename ValueType>
std::vector<int> workspace_traits<Cg<ValueType>>::vectors(const Solver&)
{
    return {r, z, p, q};
}


#define GKO_DECLARE_CG(_type) class Cg<_type>
#define GKO_DECLARE_CG_TRAITS(_type) struct workspace_traits<Cg<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_TRAITS);


}  // namespace solver
}  // namespace gko
