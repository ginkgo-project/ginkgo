// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/solver/pipe_cg.hpp"

#include <string>

#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range.hpp>

#include "core/base/dispatch_helper.hpp"
#include "core/config/solver_config.hpp"
#include "core/solver/pipe_cg_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace pipe_cg {
namespace {


GKO_REGISTER_OPERATION(initialize_1, pipe_cg::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, pipe_cg::initialize_2);
GKO_REGISTER_OPERATION(step_1, pipe_cg::step_1);
GKO_REGISTER_OPERATION(step_2, pipe_cg::step_2);


}  // anonymous namespace
}  // namespace pipe_cg


template <typename ValueType>
typename PipeCg<ValueType>::parameters_type PipeCg<ValueType>::parse(
    const config::pnode& config, const config::registry& context,
    const config::type_descriptor& td_for_child)
{
    auto params = solver::PipeCg<ValueType>::build();
    config::config_check_decorator config_check(config);
    config::common_solver_parse(params, config_check, context, td_for_child);

    return params;
}


template <typename ValueType>
std::unique_ptr<LinOp> PipeCg<ValueType>::transpose() const
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
std::unique_ptr<LinOp> PipeCg<ValueType>::conj_transpose() const
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
void PipeCg<ValueType>::apply_impl(const AbstractMultiVector* b,
                                   AbstractMultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }

    precision_dispatch<ValueType>(
        [this](auto converted_b, auto converted_x) {
            using std::swap;
            using LocalVector = matrix::MultiVector<ValueType>;

            constexpr uint8 RelativeStoppingId{1};

            auto exec = this->get_executor();
            this->setup_workspace();

            // we combine the two vectors r and w, formerly created with
            // GKO_SOLVER_VECTOR(r, converted_b);
            // GKO_SOLVER_VECTOR(w, converted_b);
            // into rw that we later slice for efficient dot product computation
            auto local_original_size =
                converted_b->template get_const_local_device_view<ValueType>()
                    .size;
            auto global_original_size = converted_b->get_size();
            dim<2> local_conjoined_size = {local_original_size[0],
                                           local_original_size[1] * 2};
            dim<2> global_conjoined_size = {global_original_size[0],
                                            local_original_size[1] * 2};

            AbstractMultiVector* rw = this->create_workspace_op_with_type_of(
                GKO_SOLVER_TRAITS::rw, converted_b, global_conjoined_size,
                local_conjoined_size);
            auto r_unique = rw->create_subview(
                local_span{0, local_original_size[0]},
                local_span{0, local_original_size[1]}, global_original_size);
            auto* r = r_unique.get();
            auto w_unique = rw->create_subview(
                local_span{0, local_original_size[0]},
                local_span{local_original_size[1],
                           local_original_size[1] + local_original_size[1]},
                global_original_size);
            auto* w = w_unique.get();

            // z now consists of two identical repeating parts: z1 and z2,
            // again, for the same reason
            GKO_SOLVER_VECTOR(z, rw);
            auto z1_unique = z->create_subview(
                local_span{0, local_original_size[0]},
                local_span{0, local_original_size[1]}, global_original_size);
            auto* z1 = z1_unique.get();
            auto z2_unique = z->create_subview(
                local_span{0, local_original_size[0]},
                local_span{local_original_size[1],
                           local_original_size[1] + local_original_size[1]},
                global_original_size);
            auto* z2 = z2_unique.get();

            GKO_SOLVER_VECTOR(p, converted_b);
            GKO_SOLVER_VECTOR(m, converted_b);
            GKO_SOLVER_VECTOR(n, converted_b);
            GKO_SOLVER_VECTOR(q, converted_b);
            GKO_SOLVER_VECTOR(f, converted_b);
            GKO_SOLVER_VECTOR(g, converted_b);

            // rho and delta become combined as well
            GKO_SOLVER_SCALAR(rhodelta, rw);
            auto rho_unique = rhodelta->create_subview(
                local_span{0, 1}, local_span{0, local_original_size[1]},
                dim<2>{1, global_original_size[1]});
            auto* rho = rho_unique.get();
            auto delta_unique = rhodelta->create_subview(
                local_span{0, 1},
                local_span{local_original_size[1],
                           local_original_size[1] + local_original_size[1]},
                dim<2>{1, global_original_size[1]});
            auto* delta = delta_unique.get();

            GKO_SOLVER_SCALAR(beta, converted_b);
            GKO_SOLVER_SCALAR(prev_rho, converted_b);

            GKO_SOLVER_ONE_MINUS_ONE();

            bool one_changed{};

            // needs to match the size of the combined rhodelta
            auto& stop_status =
                this->template create_workspace_array<stopping_status>(
                    GKO_SOLVER_TRAITS::stop, global_original_size[1]);
            auto& reduction_tmp = this->template create_workspace_array<char>(
                GKO_SOLVER_TRAITS::tmp);

            // r = b
            // prev_rho = 1.0
            exec->run(pipe_cg::make_initialize_1(
                converted_b->template get_const_local_device_view<ValueType>(),
                r->template get_local_device_view<ValueType>(),
                prev_rho->get_device_view(), stop_status));
            // r = r - Ax
            this->get_system_matrix()->apply(neg_one_op, converted_x, one_op,
                                             r);
            // z = preconditioner * r
            this->get_preconditioner()->apply(r, z1);
            // z2 = z1
            z2->copy_from(z1);
            // w = A * z
            this->get_system_matrix()->apply(z1, w);
            // m = preconditioner * w
            this->get_preconditioner()->apply(w, m);
            // n = A * m
            this->get_system_matrix()->apply(m, n);
            // merged dot products
            // rho = dot(r, z1)
            // delta = dot(w, z2)
            rw->compute_conj_dot(z, rhodelta, reduction_tmp);

            // check for an early termination
            auto stop_criterion = this->get_stop_criterion_factory()->generate(
                this->get_system_matrix(),
                std::shared_ptr<const AbstractMultiVector>(
                    converted_b, [](const AbstractMultiVector*) {}),
                converted_x, r);
            int iter = 0;
            bool all_stopped = stop_criterion->update()
                                   .num_iterations(iter)
                                   .residual(r)
                                   .implicit_sq_residual_norm(rho)
                                   .solution(converted_x)
                                   .check(RelativeStoppingId, true,
                                          &stop_status, &one_changed);
            this->template log<log::Logger::iteration_complete>(
                this, converted_b, converted_x, iter, r, nullptr, rho,
                &stop_status, all_stopped);
            if (all_stopped) {
                return;
            }

            // beta = delta
            // p = z
            // q = w
            // f = m
            // g = n
            exec->run(pipe_cg::make_initialize_2(
                p->template get_local_device_view<ValueType>(),
                q->template get_local_device_view<ValueType>(),
                f->template get_local_device_view<ValueType>(),
                g->template get_local_device_view<ValueType>(),
                beta->get_device_view(),
                z1->template get_const_local_device_view<ValueType>(),
                w->template get_const_local_device_view<ValueType>(),
                m->template get_const_local_device_view<ValueType>(),
                n->template get_const_local_device_view<ValueType>(),
                delta->get_const_device_view()));

            /* Memory movement summary:
             TODO
             */
            while (true) {
                // tmp = rho / beta
                // x = x + tmp * p
                // r = r - tmp * q
                // z = z - tmp * f
                // w = w - tmp * g
                // it's the only place where z is updated so we updated both z1
                // and z2 here
                exec->run(pipe_cg::make_step_1(
                    converted_x->template get_local_device_view<ValueType>(),
                    r->template get_local_device_view<ValueType>(),
                    z1->template get_local_device_view<ValueType>(),
                    z2->template get_local_device_view<ValueType>(),
                    w->template get_local_device_view<ValueType>(),
                    p->template get_const_local_device_view<ValueType>(),
                    q->template get_const_local_device_view<ValueType>(),
                    f->template get_const_local_device_view<ValueType>(),
                    g->template get_const_local_device_view<ValueType>(),
                    rho->get_const_device_view(), beta->get_const_device_view(),
                    stop_status));

                // m = preconditioner * w
                this->get_preconditioner()->apply(w, m);
                // n = A * m
                this->get_system_matrix()->apply(m, n);
                // prev_rho = rho
                prev_rho->copy_from(rho);
                // merged dot products
                // rho = dot(r, z1)
                // delta = dot(w, z2)
                rw->compute_conj_dot(z, rhodelta, reduction_tmp);
                // check
                ++iter;
                bool all_stopped = stop_criterion->update()
                                       .num_iterations(iter)
                                       .residual(r)
                                       .implicit_sq_residual_norm(rho)
                                       .solution(converted_x)
                                       .check(RelativeStoppingId, true,
                                              &stop_status, &one_changed);
                this->template log<log::Logger::iteration_complete>(
                    this, converted_b, converted_x, iter, r, nullptr, rho,
                    &stop_status, all_stopped);
                if (all_stopped) {
                    break;
                }

                // tmp = rho / prev_rho
                // beta = delta - |tmp|^2 * beta
                // p = z + tmp * p
                // q = w + tmp * q
                // f = m + tmp * f
                // g = n + tmp * g
                exec->run(pipe_cg::make_step_2(
                    beta->get_device_view(),
                    p->template get_local_device_view<ValueType>(),
                    q->template get_local_device_view<ValueType>(),
                    f->template get_local_device_view<ValueType>(),
                    g->template get_local_device_view<ValueType>(),
                    z1->template get_const_local_device_view<ValueType>(),
                    w->template get_const_local_device_view<ValueType>(),
                    m->template get_const_local_device_view<ValueType>(),
                    n->template get_const_local_device_view<ValueType>(),
                    prev_rho->get_const_device_view(),
                    rho->get_const_device_view(),
                    delta->get_const_device_view(), stop_status));
            }
        },
        b, x);
}


template <typename ValueType>
void PipeCg<ValueType>::apply_impl(const AbstractMultiVector* alpha,
                                   const AbstractMultiVector* b,
                                   const AbstractMultiVector* beta,
                                   AbstractMultiVector* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    LinOp::apply_impl(alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<PipeCg<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<PipeCg<ValueType>>::num_vectors(const Solver&)
{
    return 13;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<PipeCg<ValueType>>::op_names(
    const Solver&)
{
    return {
        "rw", "z",    "p",        "m",        "n",   "q",         "f",
        "g",  "beta", "rhodelta", "prev_rho", "one", "minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<PipeCg<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<PipeCg<ValueType>>::scalars(const Solver&)
{
    return {beta, rhodelta, prev_rho};
}


template <typename ValueType>
std::vector<int> workspace_traits<PipeCg<ValueType>>::vectors(const Solver&)
{
    return {rw, z, p, m, n, q, f, g};
}


#define GKO_DECLARE_PIPE_CG(ValueType) class PipeCg<ValueType>
#define GKO_DECLARE_PIPE_CG_TRAITS(ValueType) \
    struct workspace_traits<PipeCg<ValueType>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PIPE_CG_TRAITS);


}  // namespace solver
}  // namespace gko
