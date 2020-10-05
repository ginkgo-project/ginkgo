/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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
GKO_REGISTER_OPERATION(finalize, idr::finalize);
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

template <typename ValueType>
std::unique_ptr<matrix::Dense<ValueType>> interpret_as_value_type(
    matrix::Dense<to_complex<ValueType>> *source)
{
    auto exec = source->get_executor();
    auto rows = source->get_size()[0];
    auto cols = is_complex<ValueType>() ? source->get_size()[1]
                                        : 2 * source->get_size()[1];
    auto stride = is_complex<ValueType>() ? source->get_stride()
                                          : 2 * source->get_stride();
    return matrix::Dense<ValueType>::create(
        exec, gko::dim<2>{rows, cols},
        gko::Array<ValueType>::view(
            exec, rows * stride,
            reinterpret_cast<ValueType *>(source->get_values())),
        stride);
}


template <typename ValueType>
void Idr<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    using std::swap;
    using ComplexType = to_complex<ValueType>;
    using Vector = matrix::Dense<ComplexType>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;
    bool complex = true;

    auto exec = this->get_executor();

    auto one_op =
        initialize<matrix::Dense<ValueType>>({one<ValueType>()}, exec);
    auto neg_one_op =
        initialize<matrix::Dense<ValueType>>({-one<ValueType>()}, exec);
    auto complex_neg_one_op =
        initialize<Vector>({-one<to_complex<ValueType>>()}, exec);

    auto dense_b = as<matrix::Dense<ValueType>>(b);
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto complex_b = Vector::create(exec, dense_b->get_size());
    auto complex_x = Vector::create(exec, dense_x->get_size());


    constexpr uint8 RelativeStoppingId{1};

    const auto problem_size = system_matrix_->get_size()[0];
    const auto nrhs = complex_b->get_size()[1];

    auto residual = Vector::create_with_config_of(complex_b.get());
    auto v = Vector::create_with_config_of(complex_b.get());
    auto t = Vector::create_with_config_of(complex_b.get());
    auto helper = Vector::create_with_config_of(complex_b.get());

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

    // Initialization
    // m = identity
    exec->run(idr::make_initialize(
        dense_b, dense_x, complex_b.get(), complex_x.get(), m.get(),
        subspace_vectors.get(), deterministic_, &stop_status));

    // omega = 1
    exec->run(
        idr::make_fill_array(omega->get_values(), nrhs, one<ComplexType>()));

    // residual = b - Ax
    residual->copy_from(complex_b.get());
    system_matrix_->apply(neg_one_op.get(), complex_x.get(), one_op.get(),
                          residual.get());

    // g = u = 0
    exec->run(idr::make_fill_array(
        g->get_values(), problem_size * g->get_stride(), zero<ComplexType>()));
    exec->run(idr::make_fill_array(
        u->get_values(), problem_size * u->get_stride(), zero<ComplexType>()));


    /*auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(complex_b.get(), [](const
       LinOp *) {}), complex_x.get(), residual.get());*/

    int total_iter = -1;

    while (total_iter < 3) {
        ++total_iter;
        /*this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), complex_x.get());

        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual.get())
                .solution(complex_x.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }*/

        subspace_vectors->apply(residual.get(), f.get());
        // f = P^H * residual

        for (size_type k = 0; k < subspace_dim_; k++) {
            exec->run(idr::make_step_1(k, m.get(), f.get(), residual.get(),
                                       g.get(), c.get(), v.get(),
                                       &stop_status));
            // c = M \ f = (c_1, ..., c_s)^T
            // v = residual - c_k * g_k - ... - c_s * g_s

            get_preconditioner()->apply(
                interpret_as_value_type<ValueType>(v.get()).get(),
                interpret_as_value_type<ValueType>(helper.get()).get());

            exec->run(idr::make_step_2(k, omega.get(), helper.get(), c.get(),
                                       u.get(), &stop_status));
            // u_k = omega * preconditioned_vector + c_k * u_k + ... + c_s * u_s

            auto u_k = u->create_submatrix(span{0, problem_size},
                                           span{k * nrhs, (k + 1) * nrhs});

            system_matrix_->apply(u_k.get(), helper.get());
            // g_k = Au_k

            exec->run(idr::make_step_3(k, subspace_vectors.get(), g.get(),
                                       helper.get(), u.get(), m.get(), f.get(),
                                       alpha.get(), residual.get(),
                                       complex_x.get(), &stop_status));
            // for i = 1 to k - 1 do
            //     alpha = p^H_i * g_k / m_i,i
            //     g_k -= alpha * g_i
            //     u_k -= alpha * u_i
            // end for
            // for i = k to s do
            //     m_i,k = p^H_i * g_k
            // end for
            // beta = f_k / m_k,k
            // residual -= beta * g_k
            // dense_x += beta * u_k
            // f = (0,...,0,f_k+1 - beta * m_k+1,k,...,f_s - beta * m_s,k)
        }

        get_preconditioner()->apply(
            interpret_as_value_type<ValueType>(residual.get()).get(),
            interpret_as_value_type<ValueType>(helper.get()).get());
        system_matrix_->apply(helper.get(), t.get());

        t->compute_dot(residual.get(), omega.get());
        // t->compute_norm2(t_norm.get());
        t->compute_dot(t.get(), tht.get());
        residual->compute_norm2(residual_norm.get());

        exec->run(idr::make_compute_omega(
            kappa_, tht.get(), residual_norm.get(), omega.get(), &stop_status));

        t->scale(complex_neg_one_op.get());
        residual->add_scaled(omega.get(), t.get());
        complex_x.get()->add_scaled(omega.get(), helper.get());

        // omega = (t^H * residual) / (t^H * t)
        // rho = (t^H * residual) / (norm(t) * norm(residual))
        // if abs(rho) < kappa then
        //     omega *= kappa / abs(rho)
        // end if
        // residual -= omega * t
        // dense_x += omega * v
    }
    exec->run(idr::make_finalize(complex_x.get(), dense_x));
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
