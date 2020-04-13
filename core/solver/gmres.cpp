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

#include <ginkgo/core/solver/gmres.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/identity.hpp>


#include "core/solver/gmres_kernels.hpp"


namespace gko {
namespace solver {


namespace gmres {


GKO_REGISTER_OPERATION(initialize_1, gmres::initialize_1);
GKO_REGISTER_OPERATION(initialize_2, gmres::initialize_2);
GKO_REGISTER_OPERATION(step_1, gmres::step_1);
GKO_REGISTER_OPERATION(step_2, gmres::step_2);


}  // namespace gmres
// r = loops%krylov
// k = krylov
// Read: 5*ValueType*n + nnz*(2*IndexType + 2*ValueType) +
// loops_k * (ValueType*k + 9*ValueType*n + nnz*(2*IndexType + 2*ValueType) + (ValueType*k*(k + 1))/2) +
// loops * (ValueType*n + ValueType*(4*r + (2*n + 1)*(r + 1) + 7) + nnz*(2*IndexType + 2*ValueType)) +
// (ValueType*(r^2 + 3*r + 8*n))/2
// Write: 2*ValueType + 3*ValueType*k + 4*ValueType*n + 8 +
// loops_k * (ValueType + 2*ValueType*k + 6*ValueType*n + 8) +
// loops * (ValueType*(4*n + 3*r + n*r + 7)) +
// ValueType*(2*n + r)

// Notes: loops * r should be loops_k*(0 + 1 + ... + k - 1) + 0 + 1 + ... + r-1
//        = floor(loops/k) * (k - 1) * k / 2 + (r - 1) * r/ 2 (loops_r)
// Notes: loops_k is the floor(loops/k) i.e. how many does restart step activate
// Refined:
// Read: ((r^2 + 3 * r) / 2 + 9 * n + 2 * nnz) * ValueType + 2 * nnz * IndexType
// + floor(loops/k) * ((k^2 / 2 + 3 * k / 2 + 2 * nnz + 9 * n) * ValueType + 2 * nnz * IndexType)
// + loops * ((3 * n + 8 + 2 * nnz) * ValueType + 2 * nnz * IndexType)
// + loops_r * ((2 * n + 5) * ValueType)
// Write: (2 + 3 * k + 6 * n + r) * ValueType + 8
// + floor(loops/k) * ((2 * k + 6 * n + 1) * ValueType + 8)
// + loops * ((4 * n + 7) * ValueType)
// + loops_r * ((n + 3) * ValueType)

template <typename ValueType>
std::unique_ptr<LinOp> Gmres<ValueType>::transpose() const
{
    return build()
        .with_generated_preconditioner(
            share(as<Transposable>(this->get_preconditioner())->transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .with_krylov_dim(this->get_krylov_dim())
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Gmres<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_preconditioner(share(
            as<Transposable>(this->get_preconditioner())->conj_transpose()))
        .with_criteria(this->stop_criterion_factory_)
        .with_krylov_dim(this->get_krylov_dim())
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    GKO_ASSERT_IS_SQUARE_MATRIX(system_matrix_);

    using Vector = matrix::Dense<ValueType>;
    using NormVector = matrix::Dense<remove_complex<ValueType>>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto one_op = initialize<Vector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<Vector>({-one<ValueType>()}, exec);

    auto dense_b = as<const Vector>(b);
    auto dense_x = as<Vector>(x);
    auto residual = Vector::create_with_config_of(dense_b);
    auto krylov_bases = Vector::create(
        exec, dim<2>{system_matrix_->get_size()[1] * (krylov_dim_ + 1),
                     dense_b->get_size()[1]});
    std::shared_ptr<matrix::Dense<ValueType>> preconditioned_vector =
        Vector::create_with_config_of(dense_b);
    auto hessenberg = Vector::create(
        exec, dim<2>{krylov_dim_ + 1, krylov_dim_ * dense_b->get_size()[1]});
    auto givens_sin =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto givens_cos =
        Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});
    auto residual_norm_collection =
        Vector::create(exec, dim<2>{krylov_dim_ + 1, dense_b->get_size()[1]});
    auto residual_norm =
        NormVector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    Array<size_type> final_iter_nums(this->get_executor(),
                                     dense_b->get_size()[1]);
    auto y = Vector::create(exec, dim<2>{krylov_dim_, dense_b->get_size()[1]});

    bool one_changed{};
    Array<stopping_status> stop_status(this->get_executor(),
                                       dense_b->get_size()[1]);

    // Initialization
    // Read: n * ValueType
    // Write: (2 * k + n + 1) * ValueType
    exec->run(gmres::make_initialize_1(dense_b, b_norm.get(), residual.get(),
                                       givens_sin.get(), givens_cos.get(),
                                       &stop_status, krylov_dim_));
    // residual = dense_b
    // givens_sin = givens_cos = 0
    // Read: (2 * ValueType + 2 * IndexType)*nnz + 3 * n * ValueType
    // Write: n * ValueType
    system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                          residual.get());
    // residual = residual - Ax
    // Read: n * ValueType
    // Write: (1 + k + 2 * n) * ValueType + 8
    exec->run(gmres::make_initialize_2(
        residual.get(), residual_norm.get(), residual_norm_collection.get(),
        krylov_bases.get(), &final_iter_nums, krylov_dim_));
    // residual_norm = norm(residual)
    // residual_norm_collection = {residual_norm, unchanged}
    // krylov_bases(:, 1) = residual / residual_norm
    // final_iter_nums = {0, ..., 0}

    auto stop_criterion = stop_criterion_factory_->generate(
        system_matrix_, std::shared_ptr<const LinOp>(b, [](const LinOp *) {}),
        x, residual.get());

    int total_iter = -1;
    size_type restart_iter = 0;

    auto before_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(dense_x);
    auto after_preconditioner =
        matrix::Dense<ValueType>::create_with_config_of(dense_x);

    while (true) {
        ++total_iter;
        this->template log<log::Logger::iteration_complete>(
            this, total_iter, residual.get(), dense_x, residual_norm.get());
        if (stop_criterion->update()
                .num_iterations(total_iter)
                .residual(residual.get())
                .residual_norm(residual_norm.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }


        if (restart_iter == krylov_dim_) {
            // Restart
            // Read: ((k + 1) * k /2 )* ValueType + k * ValueType
            // Write: k * ValueType
            exec->run(gmres::make_step_2(residual_norm_collection.get(),
                                         krylov_bases.get(), hessenberg.get(),
                                         y.get(), before_preconditioner.get(),
                                         &final_iter_nums));
            // Solve upper triangular.
            // y = hessenberg \ residual_norm_collection
            // before_preconditioner = krylov_bases * y

            // Read: n * ValueType
            // Write: n * ValueType
            get_preconditioner()->apply(before_preconditioner.get(),
                                        after_preconditioner.get());
            // Read: 3 * n * ValueType
            // Write: n * ValueType
            dense_x->add_scaled(one_op.get(), after_preconditioner.get());
            // Solve x
            // x = x + get_preconditioner() * before_preconditioner
            // Read: n * ValueType
            // Write: n * ValueType
            residual->copy_from(dense_b);
            // residual = dense_b
            // Read: (2 * ValueType + 2 * IndexType)*nnz + 3 * n * ValueType
            // Write: n * ValueType
            system_matrix_->apply(neg_one_op.get(), dense_x, one_op.get(),
                                  residual.get());
            // residual = residual - Ax
            // Read: n * ValueType
            // Write: (1 + k + 2 * n) * ValueType + 8
            exec->run(gmres::make_initialize_2(
                residual.get(), residual_norm.get(),
                residual_norm_collection.get(), krylov_bases.get(),
                &final_iter_nums, krylov_dim_));
            // residual_norm = norm(residual)
            // residual_norm_collection = {residual_norm, unchanged}
            // krylov_bases(:, 1) = residual / residual_norm
            // final_iter_nums = {0, ..., 0}
            restart_iter = 0;
        }
        auto this_krylov = krylov_bases->create_submatrix(
            span{system_matrix_->get_size()[0] * restart_iter,
                 system_matrix_->get_size()[0] * (restart_iter + 1)},
            span{0, dense_b->get_size()[1]});

        auto next_krylov = krylov_bases->create_submatrix(
            span{system_matrix_->get_size()[0] * (restart_iter + 1),
                 system_matrix_->get_size()[0] * (restart_iter + 2)},
            span{0, dense_b->get_size()[1]});
        // Read: n * ValueType
        // Write: n * ValueType
        get_preconditioner()->apply(this_krylov.get(),
                                    preconditioned_vector.get());
        // preconditioned_vector = get_preconditioner() * this_krylov

        // Do Arnoldi and givens rotation
        // (r + 2) x 1
        auto hessenberg_iter = hessenberg->create_submatrix(
            span{0, restart_iter + 2},
            span{dense_b->get_size()[1] * restart_iter,
                 dense_b->get_size()[1] * (restart_iter + 1)});

        // Start of arnoldi
        // Read: (2 * ValueType + 2 * IndexType)*nnz
        // Write: n * ValueType
        system_matrix_->apply(preconditioned_vector.get(), next_krylov.get());
        // next_krylov = A * preconditioned_vector
        // Read: ((2 * n + 1) * (r + 1)+4*r + 3+4) * ValueType
        // Write: ((r + 1) * (1 + n) + 1 + n+2*r + 2+3) * ValueType
        exec->run(gmres::make_step_1(
            dense_b->get_size()[0], givens_sin.get(), givens_cos.get(),
            residual_norm.get(), residual_norm_collection.get(),
            krylov_bases.get(), hessenberg_iter.get(), restart_iter,
            &final_iter_nums, &stop_status));
        // final_iter_nums += 1 (unconverged)
        // R: (2 * n + 1) * (r + 1)
        // W: (r + 1) * (1 + n) + 1 + n
        // next_krylov_basis is alias for (restart_iter + 1)-th krylov_bases
        // for i in 0:restart_iter(include)
        //     hessenberg(restart_iter, i) = next_krylov_basis' *
        //         krylov_bases(:, i)
        //     next_krylov_basis  -= hessenberg(restart_iter, i) *
        //         krylov_bases(:, i)
        // end
        // hessenberg(restart_iter+1, restart_iter) = norm(next_krylov_basis)
        // next_krylov_basis /= hessenberg(restart_iter + 1, restart_iter)
        // End of arnoldi
        // R: 4*r + 3
        // W: 2*r + 2
        // Start apply givens rotation
        // for j in 0:restart_iter(exclude)
        //     temp             =  cos(j)*hessenberg(j) +
        //                         sin(j)*hessenberg(j+1)
        //     hessenberg(j+1)  = -conj(sin(j))*hessenberg(j) +
        //                         conj(cos(j))*hessenberg(j+1)
        //     hessenberg(j)    =  temp;
        // end
        // Calculate sin and cos
        // this_hess = hessenberg(restart_iter)
        // next_hess = hessenberg(restart_iter+1)
        // hypotenuse = sqrt(this_hess * this_hess + next_hess * next_hess);
        // cos(restart_iter) = conj(this_hess) / hypotenuse;
        // sin(restart_iter) = conj(next_hess) / this_hess
        // hessenberg(restart_iter)   =
        //      cos(restart_iter)*hessenberg(restart_iter) +
        //      sin(restart_iter)*hessenberg(restart_iter)
        // hessenberg(restart_iter+1) = 0
        // End apply givens rotation
        // R: 4
        // W: 3
        // Calculate residual norm
        // this_rnc = residual_norm_collection(restart_iter)
        // next_rnc = -conj(sin(restart_iter)) * this_rnc
        // residual_norm_collection(restart_iter) = cos(restart_iter) * this_rnc
        // residual_norm = abs(next_rnc)
        // residual_norm_collection(restart_iter + 1) = next_rnc

        restart_iter++;
    }

    // Solve x
    auto krylov_bases_small = krylov_bases->create_submatrix(
        span{0, system_matrix_->get_size()[0] * (restart_iter + 1)},
        span{0, dense_b->get_size()[1]});
    // r*r
    auto hessenberg_small = hessenberg->create_submatrix(
        span{0, restart_iter},
        span{0, dense_b->get_size()[1] * (restart_iter)});

    // Read: (r + 1) * r /2 * ValueType + r * ValueType
    // Write: r * ValueType
    exec->run(gmres::make_step_2(
        residual_norm_collection.get(), krylov_bases_small.get(),
        hessenberg_small.get(), y.get(), before_preconditioner.get(),
        &final_iter_nums));
    // Solve upper triangular.
    // y = hessenberg \ residual_norm_collection
    // before_preconditioner = krylov_bases * y
    // Read: n * ValueType
    // Write: n * ValueType
    get_preconditioner()->apply(before_preconditioner.get(),
                                after_preconditioner.get());
    // Read: 3 * n * ValueType
    // Write: n * ValueType
    dense_x->add_scaled(one_op.get(), after_preconditioner.get());
    // Solve x
    // x = x + get_preconditioner() * before_preconditioner
}


template <typename ValueType>
void Gmres<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                  const LinOp *residual_norm_collection,
                                  LinOp *x) const
{
    auto dense_x = as<matrix::Dense<ValueType>>(x);

    auto x_clone = dense_x->clone();
    this->apply(b, x_clone.get());
    dense_x->scale(residual_norm_collection);
    dense_x->add_scaled(alpha, x_clone.get());
}


#define GKO_DECLARE_GMRES(_type) class Gmres<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GMRES);


}  // namespace solver
}  // namespace gko
