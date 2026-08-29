// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/cg.hpp"

#include <memory>
#include <string>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/validation.hpp"
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
validation::ValidationResult is_symmetric(const LinOp* mat)
{
    using Mtx = matrix::Csr<ValueType>;
    auto exec = mat->get_executor();
    auto master = exec->get_master();
    auto cg_mtx = gko::copy_and_convert_to<Mtx>(exec, mat);

    if (cg_mtx->get_size()[0] != cg_mtx->get_size()[1]) {
        return {false, "Matrix is not square."};
    }

    auto trans_cg_mtx =
        gko::copy_and_convert_to<Mtx>(exec, cg_mtx.get()->transpose().get());

    auto host_cg_mtx = Mtx::create(master);
    host_cg_mtx->copy_from(cg_mtx.get());

    auto host_trans_cg_mtx = Mtx::create(master);
    host_trans_cg_mtx->copy_from(trans_cg_mtx.get());

    host_cg_mtx->sort_by_column_index();
    host_trans_cg_mtx->sort_by_column_index();

    auto row_ptrs = host_cg_mtx->get_const_row_ptrs();
    auto col_idxs = host_cg_mtx->get_const_col_idxs();
    auto values = host_cg_mtx->get_const_values();

    auto trans_row_ptrs = host_trans_cg_mtx->get_const_row_ptrs();
    auto trans_col_idxs = host_trans_cg_mtx->get_const_col_idxs();
    auto trans_values = host_trans_cg_mtx->get_const_values();

    auto nnz = host_cg_mtx->get_num_stored_elements();

    for (size_type i = 0; i < host_cg_mtx->get_size()[0]; ++i) {
        if (row_ptrs[i] != trans_row_ptrs[i]) {
            return {false, "Row pointers at index " + std::to_string(i)};
        }
    }

    for (size_type i = 0; i < nnz; ++i) {
        if (col_idxs[i] != trans_col_idxs[i]) {
            return {false, "Column index at index " + std::to_string(i)};
        }
        if (values[i] != trans_values[i]) {
            return {false, "Value at index " + std::to_string(i)};
        }
    }

    return {true, ""};
}


template <typename ValueType>
void Cg<ValueType>::validate_data() const
{
    validation::validate_system_matrix<ValueType, int32>(
        this->get_system_matrix());
    GKO_VALIDATE(is_symmetric<ValueType>(this->get_system_matrix().get()),
                 "The system matrix is not symmetric.");
}


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
        gko::detail::get_local(dense_b)->get_const_device_view(),
        gko::detail::get_local(r)->get_device_view(),
        gko::detail::get_local(z)->get_device_view(),
        gko::detail::get_local(p)->get_device_view(),
        gko::detail::get_local(q)->get_device_view(),
        prev_rho->get_device_view(), rho->get_device_view(), stop_status));

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
        exec->run(
            cg::make_step_1(gko::detail::get_local(p)->get_device_view(),
                            gko::detail::get_local(z)->get_const_device_view(),
                            rho->get_const_device_view(),
                            prev_rho->get_const_device_view(), stop_status));
        // q = A * p
        this->get_system_matrix()->apply(p, q);
        // beta = dot(p, q)
        p->compute_conj_dot(q, beta, reduction_tmp);
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(
            cg::make_step_2(gko::detail::get_local(dense_x)->get_device_view(),
                            gko::detail::get_local(r)->get_device_view(),
                            gko::detail::get_local(p)->get_const_device_view(),
                            gko::detail::get_local(q)->get_const_device_view(),
                            beta->get_const_device_view(),
                            rho->get_const_device_view(), stop_status));
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


#define GKO_DECLARE_CG(ValueType) class Cg<ValueType>
#define GKO_DECLARE_CG_TRAITS(ValueType) struct workspace_traits<Cg<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG_TRAITS);


}  // namespace solver
}  // namespace gko
