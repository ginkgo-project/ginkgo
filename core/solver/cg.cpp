// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/cg.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/mtx_io.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/factorization/lu.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/config/config_helper.hpp"
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
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

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
            if (n <= 4) {
                dense_eigs->fill(one<ValueType>());
            } else {
                // std::ofstream out_alpha{"alpha_" + std::to_string(n-4) +
                // ".mtx"}; std::ofstream out_beta{"beta_" + std::to_string(n-4)
                // + ".mtx"};
                gko::matrix_data<ValueType, int> t_data(dim<2>{n - 4, n - 4});
                t_data.nonzeros.emplace_back(0, 0, est_beta[1] / est_alpha[1]);
                // out_alpha << est_alpha[1] << std::endl;
                // out_beta << est_beta[1] << std::endl;
                for (size_type i = 0; i < n - 5; i++) {
                    // out_alpha << est_alpha[i + 2] << std::endl;
                    // out_beta << est_beta[i + 2] << std::endl;
                    t_data.nonzeros.emplace_back(
                        i, i, one<ValueType>() / est_alpha[i + 2]);
                    t_data.nonzeros.emplace_back(
                        i + 1, i, sqrt(est_beta[i + 2]) / est_alpha[i + 2]);
                    t_data.nonzeros.emplace_back(
                        i, i + 1, sqrt(est_beta[i + 2]) / est_alpha[i + 2]);
                    t_data.nonzeros.emplace_back(
                        i + 1, i + 1, est_beta[i + 2] / est_alpha[i + 2]);
                }
                // out_alpha << est_alpha[n - 3] << std::endl;
                t_data.nonzeros.emplace_back(
                    n - 5, n - 5, one<ValueType>() / est_alpha[n - 3]);
                t_data.sum_duplicates();
                auto t = share(matrix::Csr<ValueType, int>::create(exec));
                t->read(t_data);
                auto fact = share(
                    experimental::factorization::Lu<ValueType, int>::build()
                        .on(exec)
                        ->generate(t)
                        ->unpack());
                auto diag = fact->get_upper_factor()->extract_diagonal();
                auto host_diag = clone(exec->get_master(), diag);
                // std::ofstream out{"diag_" +
                // std::to_string(diag->get_size()[0]) + ".mtx"};
                // gko::write(out, t);
                ValueType min_eig = 1e9;
                ValueType max_eig = 0;
                for (size_type i = 0; i < diag->get_size()[0]; i++) {
                    min_eig = min(abs(min_eig),
                                  abs(host_diag->get_const_values()[i]));
                    max_eig = max(abs(max_eig),
                                  abs(host_diag->get_const_values()[i]));
                }
                gko::detail::get_local(dense_eigs)->at(0, 0) = min_eig;
                gko::detail::get_local(dense_eigs)->at(1, 0) = max_eig;
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
        auto oldalpha = host_rho->at(0, 0) / host_beta->at(0, 0);
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
