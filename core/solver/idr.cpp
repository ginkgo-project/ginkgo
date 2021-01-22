/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include <ginkgo/core/solver/idr.hpp>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/components/fill_array.hpp"
#include "core/solver/idr_kernels.hpp"


namespace gko {
namespace solver {
namespace idr {


GKO_REGISTER_OPERATION(initialize, idr::initialize);
GKO_REGISTER_OPERATION(step_1, idr::step_1);
GKO_REGISTER_OPERATION(step_2, idr::step_2);
GKO_REGISTER_OPERATION(step_3, idr::step_3);
GKO_REGISTER_OPERATION(compute_omega, idr::compute_omega);
GKO_REGISTER_OPERATION(fill_array, components::fill_array);


}  // namespace idr


template <typename ValueType>
std::unique_ptr<LinOp> Idr<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
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
        .with_criteria(this->stop_criterion_factory_)
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}

// s is subspace vector size
// Read: n * SubspaceType + (2n + 2) * SubspaceType + matrix_storage
// + loops * (
//   (s * n + n) * SubspaceType
//   + loops_s * ((s^2/2 + 9s/2 - 3k + (3s + 2k + 10)n + 2) * SubspaceType + precond_storage + matrix_storage)
//   + (11n + 5) * SubspaceType + ValueType + matrix_storage + precond_storage
// )
// Write: (s^2 + 3n + 1) * SubspaceType + n * SubspaceType
// + loops * (
//   s * SubspaceType
//   + loops_s * ((3nk + 7n + 3s - k) * SubspaceType)
//   + (5n + 4) * SubspaceType + ValueType
// )

// loops_s * k = 0 + 1 + 2 + ... + (s-1) = (s-1) * s/2 (loops_sk), others is s
// Read: n * SubspaceType + (2n + 2) * SubspaceType + matrix_storage
// + loops * (
//   (s * n + n) * SubspaceType
//   + s * ((s^2/2 + 9s/2 + (3s + 10)n + 2) * SubspaceType + precond_storage + matrix_storage)
//   + loops_sk * (2n - 3) * SubspaceType
//   + (11n + 5) * SubspaceType + ValueType + matrix_storage + precond_storage
// )
// Write: (s^2 + 3n + 1) * SubspaceType + n * SubspaceType
// + loops * (
//   s * SubspaceType
//   + s * ((7n + 3s) * SubspaceType)
//   + loops_sk * (3n - 1) * SubspaceType
//   + (5n + 4) * SubspaceType + ValueType
// )
// FLOPS: 2nnz + n + loops * (2 * n * s + s^3 + 11ns/2 + 2nnz*s + 11ns^2/2 + s^2 + 2 * nnz + n + n - 1 +n + n - 1 +n + n - 1 +6+n+4n)
// = 2nnz + n + loops * (s^3 + s^2 + 15ns/2 + 2nnz*s + 11ns^2/2 + 2 * nnz + 11n + 3)
template <typename ValueType>
template <typename SubspaceType>
void Idr<ValueType>::iterate(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using Vector = matrix::Dense<SubspaceType>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;

    auto exec = this->get_executor();

    auto one_op =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    auto neg_one_op =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    auto subspace_neg_one_op = initialize<Vector>({-one<SubspaceType>()}, exec);

    auto dense_b = as<Vector>(b);
    auto dense_x = as<Vector>(x);

    constexpr uint8 RelativeStoppingId{1};

    const auto problem_size = system_matrix_->get_size()[0];
    const auto nrhs = dense_b->get_size()[1];

    auto residual = Vector::create_with_config_of(dense_b);
    auto v = Vector::create_with_config_of(dense_b);
    auto t = Vector::create_with_config_of(dense_b);
    auto helper = Vector::create_with_config_of(dense_b);

    auto m =
        Vector::create(exec, gko::dim<2>{subspace_dim_, subspace_dim_ * nrhs});

    auto g =
        Vector::create(exec, gko::dim<2>{problem_size, subspace_dim_ * nrhs});
    auto u =
        Vector::create(exec, gko::dim<2>{problem_size, subspace_dim_ * nrhs});

    auto f = Vector::create(exec, gko::dim<2>{subspace_dim_, nrhs});
    auto c = Vector::create(exec, gko::dim<2>{subspace_dim_, nrhs});

    auto omega = Vector::create(exec, gko::dim<2>{1, nrhs});
    auto residual_norm = NormVector::create(exec, dim<2>{1, nrhs});
    auto tht = Vector::create(exec, dim<2>{1, nrhs});
    auto t_norm = NormVector::create(exec, dim<2>{1, nrhs});
    auto alpha = Vector::create(exec, gko::dim<2>{1, nrhs});

    bool one_changed{};
    Array<stopping_status> stop_status(exec, nrhs);

    // The dense matrix containing the randomly generated subspace vectors.
    // Stored in column major order and complex conjugated. So, if the
    // matrix containing the subspace vectors in row major order is called P,
    // subspace_vectors actually contains P^H.
    auto subspace_vectors =
        Vector::create(exec, gko::dim<2>(subspace_dim_, problem_size));

    // Write: s * s * SubspaceType
    // Initialization
    // m = identity
    exec->run(idr::make_initialize(nrhs, m.get(), subspace_vectors.get(),
                                   deterministic_, &stop_status));

    // Write: SubspaceType
    // omega = 1
    exec->run(
        idr::make_fill_array(omega->get_values(), nrhs, one<SubspaceType>()));

    // Read: n * SubspaceType
    // Write: n * SubspaceType
    // residual = b - Ax
    residual->copy_from(dense_b);
    // Real
    // Read: (2n+2) * SubspaceType + matrix_storage
    // Write: n * SubspaceType
    // FLOPS: 2*nnz + n
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          residual.get());

    // Write: 2n * SubspaceType
    // g = u = 0
    exec->run(idr::make_fill_array(
        g->get_values(), problem_size * g->get_stride(), zero<SubspaceType>()));
    exec->run(idr::make_fill_array(
        u->get_values(), problem_size * u->get_stride(), zero<SubspaceType>()));


    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_,
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp *) {}), dense_x,
        residual.get());

    int total_iter = -1;

    while (true) {
        ++total_iter;
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), dense_x);

        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }
        // s * n x n = s
        // Read: (s * n + n) * SubspaceType
        // Write: s * SubspaceType
        // FLOPS: 2 * n * s
        subspace_vectors->apply(residual.get(), f.get());
        // f = P^H * residual
        // TOTAL FLOPS:
        // = s^3 - s + 6ns + 2nnz*s + 5ns^2 + 2s^2 + (n-2) * s(s-1)/2
        // = s^3 - s + 6ns + 2nnz*s + 5ns^2 + 2s^2 + ns^2/2 - ns/2 + s - s^2
        // = s^3 + 11ns/2 + 2nnz*s + 11ns^2/2 + s^2
        // For each k FLOPS:
        // 2n * (s - k) + s^2 - 1 + n + 2n * (s-k) + 2nnz + 6*n*k + (s - k) * n + 5n + 2 * (s - k)
        // = s^2 - 1 + 6n + 2nnz +  6*n*k  + (s - k) (5n + 2)
        // = s^2 - 1 + 6n + 2nnz + 5ns + 2s + nk - 2k
        for (size_type k = 0; k < subspace_dim_; k++) {
            // Read: ((s + 1) * s /2 )* SubspaceType + s * SubspaceType + (n+1)(s-k) * SubspaceType + n * SubspaceType
            // Write: s * SubspaceType + n * SubspaceType
            // FLOPS: 2n * (s - k) + s^2 - 1
            exec->run(idr::make_step_1(nrhs, k, m.get(), f.get(),
                                       residual.get(), g.get(), c.get(),
                                       v.get(), &stop_status));
            // c = M \ f = (c_1, ..., c_s)^T
            // ---
            // v = residual - c_k * g_k - ... - c_s * g_s

            // Note: Identity<ValueType> directly copy vector without checking type
            // Read: n * SubspaceType + precond_storage
            // Write: n * SubspaceType
            get_preconditioner()->apply(v.get(), helper.get());

            // Read: (n + n * (s-k) + (s - k) + 1) * SubspaceType
            // Write: n * SubspaceType
            // FLOPS: n + 2n(s-k)
            exec->run(idr::make_step_2(nrhs, k, omega.get(), helper.get(),
                                       c.get(), u.get(), &stop_status));
            // u_k = omega * preconditioned_vector + c_k * u_k + ... + c_s * u_s

            auto u_k = u->create_submatrix(span{0, problem_size},
                                           span{k * nrhs, (k + 1) * nrhs});
            // Real
            // Read: n * SubspaceType + matrix_storage
            // Write: n * SubspaceType
            // FLOPS: 2 * nnz
            system_matrix_->apply(u_k.get(), helper.get());
            // g_k = Au_k
            // Read: (4nk + ns + 6n + s - k - 1) * SubspaceType
            // Write: (3nk + 3n + 2s - k) * SubspaceType
            exec->run(idr::make_step_3(nrhs, k, subspace_vectors.get(), g.get(),
                                       helper.get(), u.get(), m.get(), f.get(),
                                       alpha.get(), residual.get(), dense_x,
                                       &stop_status));
            // R: k * (2n + 2n + n) * SubspaceType + n * SubsparceType
            // W: k * (1 + 2n + n) * SubspaceType + n * SubsparceType
            // FLOPS: 6*n*k
            // for i = 1 to k - 1 do
            //     alpha = p^H_i * g_k / m_i,i
            //     ---
            //     g_k -= alpha * g_i
            //     u_k -= alpha * u_i
            // end for
            // update g_k to g_(k) copy back the vector to matrix
            // ---
            // R: (s - k + 1) * n * SubspaceType
            // W: (s-k) * SubspaceType
            // FLOPS: (s - k) * n
            // for i = k to s do
            //     m_i,k = p^H_i * g_k
            // end for
            // ---
            // R: (2n + 2n + (s - k - 1)) * SubspaceType
            // W: (n + n + (s - k - 1) + 1) * SubspaceType
            // FLOPS: 5n + 2 * (s - k)
            // beta = f_k / m_k,k
            // residual -= beta * g_k
            // dense_x += beta * u_k
            // f = (0,...,0,f_k+1 - beta * m_k+1,k,...,f_s - beta * m_s,k)
        }
        // Read: n * SubspaceType + precond_storage
        // Write: n * SubspaceType
        get_preconditioner()->apply(residual.get(), helper.get());
        // Real
        // Read: n * SubspaceType + matrix_storage
        // Write: n * SubspaceType
        // FLOPS: 2 * nnz
        system_matrix_->apply(helper.get(), t.get());
        // Read: 2n * SubspaceType
        // Write: SubspaceType
        // FLOPS: n + n - 1
        t->compute_dot(residual.get(), omega.get());
        // Read: n * SubspaceType
        // Write: SubspaceType
        // FLOPS: n + n - 1
        t->compute_dot(t.get(), tht.get());
        // Read: n * SubspaceType
        // Real
        // Write: ValueType
        // FLOPS: n + n - 1
        residual->compute_norm2(residual_norm.get());
        // Real
        // Read: 2 * SubspaceType + ValueType
        // Write: 2 * SubspaceType
        // FLOPS: 6
        exec->run(idr::make_compute_omega(nrhs, kappa_, tht.get(),
                                          residual_norm.get(), omega.get(),
                                          &stop_status));
        // Read: (n + 1) * SubspaceType
        // Write: n * SubspaceType
        // FLOPS: n
        t->scale(subspace_neg_one_op.get());
        // Read: (2n + 1 + 2n + 1) * SubspaceType
        // Write: (n + n) * SubspaceType
        // FLOPS: 4n
        residual->add_scaled(omega.get(), t.get());
        dense_x->add_scaled(omega.get(), helper.get());

        // omega = (t^H * residual) / (t^H * t)
        // rho = (t^H * residual) / (norm(t) * norm(residual))
        // if abs(rho) < kappa then
        //     omega *= kappa / abs(rho)
        // end if
        // residual -= omega * t
        // dense_x += omega * v
    }
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    // If ValueType is complex, the subspace matrix P will be complex anyway.
    if (!is_complex<ValueType>() && complex_subspace_) {
        auto dense_b = as<matrix::Dense<ValueType>>(b);
        auto dense_x = as<matrix::Dense<ValueType>>(x);
        auto complex_b = dense_b->make_complex();
        auto complex_x = dense_x->make_complex();
        this->iterate<to_complex<ValueType>>(complex_b.get(), complex_x.get());
        complex_x->get_real(
            dynamic_cast<matrix::Dense<remove_complex<ValueType>> *>(dense_x));
    } else {
        this->iterate<ValueType>(b, x);
    }
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                const LinOp *beta, LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_IDR(_type) class Idr<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDR);


}  // namespace solver
}  // namespace gko
