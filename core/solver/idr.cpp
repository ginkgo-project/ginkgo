// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/solver/idr.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/solver/solver_base.hpp>


#include "core/distributed/helpers.hpp"
#include "core/solver/idr_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace idr {
namespace {


GKO_REGISTER_OPERATION(initialize, idr::initialize);
GKO_REGISTER_OPERATION(step_1, idr::step_1);
GKO_REGISTER_OPERATION(step_2, idr::step_2);
GKO_REGISTER_OPERATION(step_3, idr::step_3);
GKO_REGISTER_OPERATION(compute_omega, idr::compute_omega);


}  // anonymous namespace
}  // namespace idr


template <typename ValueType>
std::unique_ptr<LinOp> Idr<ValueType>::transpose() const
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
std::unique_ptr<LinOp> Idr<ValueType>::conj_transpose() const
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
template <typename VectorType>
void Idr<ValueType>::iterate(const VectorType* dense_b,
                             VectorType* dense_x) const
{
    using std::swap;
    using SubspaceType = typename VectorType::value_type;
    using Vector = matrix::Dense<SubspaceType>;
    using AbsType = remove_complex<ValueType>;
    using ws = workspace_traits<Idr>;

    auto exec = this->get_executor();
    this->setup_workspace();

    constexpr uint8 RelativeStoppingId{1};

    const auto problem_size = this->get_size()[0];
    const auto nrhs = dense_b->get_size()[1];
    const auto subspace_dim = this->get_subspace_dim();
    const auto is_deterministic = this->get_deterministic();
    const auto kappa = this->get_kappa();

    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(v, dense_b);
    GKO_SOLVER_VECTOR(t, dense_b);
    GKO_SOLVER_VECTOR(helper, dense_b);

    auto m = this->template create_workspace_op<Vector>(
        ws::m, gko::dim<2>{subspace_dim, subspace_dim * nrhs});

    auto g = this->template create_workspace_op<Vector>(
        ws::g, gko::dim<2>{problem_size, subspace_dim * nrhs});
    auto u = this->template create_workspace_op<Vector>(
        ws::u, gko::dim<2>{problem_size, subspace_dim * nrhs});

    auto f = this->template create_workspace_op<Vector>(
        ws::f, gko::dim<2>{subspace_dim, nrhs});
    auto c = this->template create_workspace_op<Vector>(
        ws::c, gko::dim<2>{subspace_dim, nrhs});

    auto omega =
        this->template create_workspace_scalar<SubspaceType>(ws::omega, nrhs);
    auto residual_norm = this->template create_workspace_scalar<AbsType>(
        ws::residual_norm, nrhs);
    auto tht =
        this->template create_workspace_scalar<SubspaceType>(ws::tht, nrhs);
    auto alpha =
        this->template create_workspace_scalar<SubspaceType>(ws::alpha, nrhs);

    // The dense matrix containing the randomly generated subspace vectors.
    // Stored in column major order and complex conjugated. So, if the
    // matrix containing the subspace vectors in row major order is called P,
    // subspace_vectors actually contains P^H.
    auto subspace_vectors = this->template create_workspace_op<Vector>(
        ws::subspace, gko::dim<2>(subspace_dim, problem_size));

    GKO_SOLVER_ONE_MINUS_ONE();
    auto subspace_neg_one_op =
        this->template create_workspace_scalar<SubspaceType>(
            ws::subspace_minus_one, 1);
    subspace_neg_one_op->fill(-one<SubspaceType>());

    bool one_changed{};
    GKO_SOLVER_STOP_REDUCTION_ARRAYS();

    // Initialization
    // m = identity
    if (is_deterministic) {
        auto subspace_vectors_data = matrix_data<SubspaceType>(
            subspace_vectors->get_size(), std::normal_distribution<>(0.0, 1.0),
            std::default_random_engine(15));
        subspace_vectors->read(subspace_vectors_data);
    }
    exec->run(idr::make_initialize(nrhs, gko::detail::get_local(m),
                                   gko::detail::get_local(subspace_vectors),
                                   is_deterministic, &stop_status));

    // omega = 1
    omega->fill(one<SubspaceType>());

    // residual = b - Ax
    residual->copy_from(dense_b);
    this->get_system_matrix()->apply(neg_one_op, dense_x, one_op, residual);
    residual->compute_norm2(residual_norm, reduction_tmp);

    // g = u = 0
    g->fill(zero<SubspaceType>());
    u->fill(zero<SubspaceType>());

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        residual);

    int total_iter = -1;

    /* Memory movement summary for iteration with subspace dimension s
     * Per iteration:
     * (11/2s^2+31/2s+18)n * values + (s+1) * matrix/preconditioner storage
     * (s+1)x SpMV:                2(s+1)n * values + (s+1) * storage
     * (s+1)x Preconditioner:      2(s+1)n * values + (s+1) * storage
     * 1x multidot (gemv)           (s+1)n
     * sx step 1 (fused axpys) s(s/2+5/2)n = sum k=[0,s) of (s-k+2)n
     * sx step 2 (fused axpys) s(s/2+5/2)n = sum k=[0,s) of (s-k+2)n
     * sx step 3:            s(9/2s+11/2)n = sum k=[0,s) of (8k+2+s-k+1+6)n
     *       1x orthogonalize g+u      (8k+2)n in iteration k (0-based)
     *       1x multidot (gemv)       (s-k+1)n in iteration k (0-based)
     *       2x axpy                        6n
     * 1x dot                           2n
     * 2x norm2                         2n
     * 1x scale                         2n
     * 2x axpy                          6n
     * 1x norm2 residual                 n
     */
    while (true) {
        ++total_iter;

        bool all_stopped =
            stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual)
                .residual_norm(residual_norm)
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed);
        this->template log<log::Logger::iteration_complete>(
            this, dense_b, dense_x, total_iter, residual, nullptr, nullptr,
            &stop_status, all_stopped);
        if (all_stopped) {
            break;
        }

        // f = P^H * residual
        subspace_vectors->apply(residual, f);

        for (size_type k = 0; k < subspace_dim; k++) {
            // c = M \ f = (c_1, ..., c_s)^T
            // v = residual - sum i=[k,s) of (c_i * g_i)
            exec->run(idr::make_step_1(
                nrhs, k, gko::detail::get_local(m), gko::detail::get_local(f),
                gko::detail::get_local(residual), gko::detail::get_local(g),
                gko::detail::get_local(c), gko::detail::get_local(v),
                &stop_status));

            this->get_preconditioner()->apply(v, helper);

            // u_k = omega * precond_vector + sum i=[k,s) of (c_i * u_i)
            exec->run(idr::make_step_2(
                nrhs, k, gko::detail::get_local(omega),
                gko::detail::get_local(helper), gko::detail::get_local(c),
                gko::detail::get_local(u), &stop_status));

            auto u_k = u->create_submatrix(span{0, problem_size},
                                           span{k * nrhs, (k + 1) * nrhs});

            // g_k = Au_k
            this->get_system_matrix()->apply(u_k, helper);

            // for i = [0,k)
            //     alpha = p^H_i * g_k / m_i,i
            //     g_k -= alpha * g_i
            //     u_k -= alpha * u_i
            // end for
            // store g_k to g
            // for i = [k,s)
            //     m_i,k = p^H_i * g_k
            // end for
            // beta = f_k / m_k,k
            // residual -= beta * g_k
            // dense_x += beta * u_k
            // f = (0,...,0,f_k+1 - beta * m_k+1,k,...,f_s-1 - beta * m_s-1,k)
            exec->run(idr::make_step_3(
                nrhs, k, gko::detail::get_local(subspace_vectors),
                gko::detail::get_local(g), gko::detail::get_local(helper),
                gko::detail::get_local(u), gko::detail::get_local(m),
                gko::detail::get_local(f), gko::detail::get_local(alpha),
                gko::detail::get_local(residual),
                gko::detail::get_local(dense_x), &stop_status));
        }

        this->get_preconditioner()->apply(residual, helper);
        this->get_system_matrix()->apply(helper, t);

        t->compute_conj_dot(residual, omega, reduction_tmp);
        t->compute_conj_dot(t, tht, reduction_tmp);
        residual->compute_norm2(residual_norm, reduction_tmp);

        // omega = (t^H * residual) / (t^H * t)
        // rho = (t^H * residual) / (norm(t) * norm(residual))
        // if abs(rho) < kappa then
        //     omega *= kappa / abs(rho)
        // end if
        // residual -= omega * t
        // dense_x += omega * v
        exec->run(idr::make_compute_omega(
            nrhs, kappa, gko::detail::get_local(tht),
            gko::detail::get_local(residual_norm),
            gko::detail::get_local(omega), &stop_status));

        t->scale(subspace_neg_one_op);
        residual->add_scaled(omega, t);
        dense_x->add_scaled(omega, helper);
    }
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            // If ValueType is complex, the subspace matrix P will be complex
            // anyway.
            if (!is_complex<ValueType>() && this->get_complex_subspace()) {
                auto complex_b = dense_b->make_complex();
                auto complex_x = dense_x->make_complex();
                this->iterate(complex_b.get(), complex_x.get());
                complex_x->get_real(
                    dynamic_cast<matrix::Dense<remove_complex<ValueType>>*>(
                        dense_x));
            } else {
                this->iterate(dense_b, dense_x);
            }
        },
        b, x);
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                const LinOp* beta, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
int workspace_traits<Idr<ValueType>>::num_arrays(const Solver&)
{
    return 2;
}


template <typename ValueType>
int workspace_traits<Idr<ValueType>>::num_vectors(const Solver&)
{
    return 17;
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Idr<ValueType>>::op_names(
    const Solver&)
{
    return {
        "residual",
        "v",
        "t",
        "helper",
        "m",
        "g",
        "u",
        "subspace",
        "f",
        "c",
        "omega",
        "residual_norm",
        "tht",
        "alpha",
        "one",
        "minus_one",
        "subspace_minus_one",
    };
}


template <typename ValueType>
std::vector<std::string> workspace_traits<Idr<ValueType>>::array_names(
    const Solver&)
{
    return {"stop", "tmp"};
}


template <typename ValueType>
std::vector<int> workspace_traits<Idr<ValueType>>::scalars(const Solver&)
{
    return {omega, tht, alpha};
}


template <typename ValueType>
std::vector<int> workspace_traits<Idr<ValueType>>::vectors(const Solver&)
{
    return {residual, v, t, helper, m, g, u, subspace, f, c};
}


#define GKO_DECLARE_IDR(_type) class Idr<_type>
#define GKO_DECLARE_IDR_TRAITS(_type) struct workspace_traits<Idr<_type>>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR_TRAITS);


}  // namespace solver
}  // namespace gko
