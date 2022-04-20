/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <ginkgo/core/solver/ir.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/distributed/matrix.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/handle_guard.hpp"
#include "core/distributed/helpers.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "core/solver/ir_kernels.hpp"


namespace gko {
namespace solver {
namespace ir {
namespace {


GKO_REGISTER_ASYNC_OPERATION(initialize, ir::initialize);
GKO_REGISTER_ASYNC_OPERATION(copy, dense::copy);


}  // anonymous namespace
}  // namespace ir


template <typename ValueType>
std::unique_ptr<LinOp> Ir<ValueType>::transpose() const
{
    return build()
        .with_generated_solver(
            share(as<Transposable>(this->get_solver())->transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_relaxation_factor(parameters_.relaxation_factor)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType>
std::unique_ptr<LinOp> Ir<ValueType>::conj_transpose() const
{
    return build()
        .with_generated_solver(
            share(as<Transposable>(this->get_solver())->conj_transpose()))
        .with_criteria(this->get_stop_criterion_factory())
        .with_relaxation_factor(conj(parameters_.relaxation_factor))
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType>
void Ir<ValueType>::set_solver(std::shared_ptr<const LinOp> new_solver)
{
    auto exec = this->get_executor();
    if (new_solver) {
        GKO_ASSERT_EQUAL_DIMENSIONS(new_solver, this);
        GKO_ASSERT_IS_SQUARE_MATRIX(new_solver);
        if (new_solver->get_executor() != exec) {
            new_solver = gko::clone(exec, new_solver);
        }
    }
    solver_ = new_solver;
}


template <typename ValueType>
void Ir<ValueType>::set_relaxation_factor(
    std::shared_ptr<const matrix::Dense<ValueType>> new_factor)
{
    auto exec = this->get_executor();
    if (new_factor && new_factor->get_executor() != exec) {
        new_factor = gko::clone(exec, new_factor);
    }
    relaxation_factor_ = new_factor;
}


template <typename ValueType>
Ir<ValueType>& Ir<ValueType>::operator=(const Ir& other)
{
    if (&other != this) {
        EnableLinOp<Ir>::operator=(other);
        EnableSolverBase<Ir>::operator=(other);
        EnableIterativeBase<Ir>::operator=(other);
        parameters_ = other.parameters_;
        if (this->get_system_matrix()) {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType>
Ir<ValueType>& Ir<ValueType>::operator=(Ir&& other)
{
    if (&other != this) {
        EnableLinOp<Ir>::operator=(std::move(other));
        EnableSolverBase<Ir>::operator=(std::move(other));
        EnableIterativeBase<Ir>::operator=(std::move(other));
        parameters_ = other.parameters_;
        other.set_solver(nullptr);
        other.set_relaxation_factor(nullptr);
        if (this->get_system_matrix()) {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType>
void Ir<ValueType>::generate()
{
    GKO_ASSERT_IS_SQUARE_MATRIX(this->get_system_matrix());
    if (parameters_.generated_solver) {
        this->set_solver(parameters_.generated_solver);
        GKO_ASSERT_EQUAL_DIMENSIONS(this->get_solver(), this);
    } else if (parameters_.solver) {
        // TODO overhead of re-generation when copying the solver
        this->set_solver(
            parameters_.solver->generate(this->get_system_matrix()));
    } else {
        this->set_solver(matrix::Identity<ValueType>::create(
            this->get_executor(), this->get_size()));
    }
    this->relaxation_factor_ = gko::initialize<matrix::Dense<ValueType>>(
        {parameters_.relaxation_factor}, this->get_executor());
    this->one_op_ = initialize<matrix::Dense<ValueType>>({one<ValueType>()},
                                                         this->get_executor());
    this->neg_one_op_ = initialize<matrix::Dense<ValueType>>(
        {-one<ValueType>()}, this->get_executor());
    this->host_storage_ = gko::Array<bool>(
        this->get_executor()->get_master(), 2,
        this->get_executor()->get_mem_space()->template pinned_host_alloc<bool>(
            2),
        memory_space_pinned_host_deleter<bool[]>(
            this->get_executor()->get_mem_space()));
    this->host_storage_.fill(false);
    auto num_rows = this->get_system_matrix()->get_size()[0];
    auto nrhs = this->get_parameters().num_rhs;
    this->workspace_ = gko::Array<ValueType>(this->get_executor(),
                                             num_rows * nrhs * num_aux_vecs);
    this->real_workspace_ =
        gko::Array<remove_complex<ValueType>>(this->get_executor(), nrhs * 2);
    this->stop_status_ =
        gko::Array<stopping_status>(this->get_executor(), nrhs);
    this->device_storage_ =
        std::make_shared<Array<bool>>(this->get_executor(), 2);
}


template <typename ValueType>
Ir<ValueType>::Ir(const Ir& other) : Ir(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Ir<ValueType>::Ir(Ir&& other) : Ir(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    auto handle = this->get_executor()->get_default_exec_stream();
    precision_dispatch_real_complex_distributed<ValueType>(
        [this, handle](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x, handle)->wait();
        },
        b, x);
}


template <typename ValueType>
std::shared_ptr<AsyncHandle> Ir<ValueType>::apply_impl(
    const LinOp* b, LinOp* x, std::shared_ptr<AsyncHandle> handle) const
{
    return async_precision_dispatch_real_complex_distributed<ValueType>(
        [this, handle](auto dense_b, auto dense_x) {
            return this->apply_dense_impl(dense_b, dense_x, handle);
        },
        b, x);
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* b, LinOp* x,
                               const OverlapMask& wmask) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this, wmask](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x, wmask);
        },
        b, x);
}


template <typename ValueType>
template <typename VectorType>
std::shared_ptr<AsyncHandle> Ir<ValueType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x,
    std::shared_ptr<AsyncHandle> handle) const
{
    using LocalVector = matrix::Dense<ValueType>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();
    handle_guard hg{exec, handle};
    auto num_rhs = dense_b->get_size()[1];
    auto b_stride = detail::get_local(dense_b)->get_stride();
    auto num_rows = detail::get_local(dense_b)->get_size()[0];
    if (this->get_parameters().num_rhs < b_stride) {
        this->workspace_ = gko::Array<ValueType>(
            this->get_executor(), num_rows * b_stride * num_aux_vecs);
        this->real_workspace_ = gko::Array<remove_complex<ValueType>>(
            this->get_executor(), b_stride * 2);
        this->stop_status_ =
            gko::Array<stopping_status>(this->get_executor(), b_stride);
    }
    int offset = 0;

    auto residual = detail::create_with_same_size_from_view(this->workspace_,
                                                            offset, dense_b);
    offset += num_rows * b_stride;
    auto inner_solution = detail::create_with_same_size_from_view(
        this->workspace_, offset, dense_b);
    offset = 0;
    auto st_tau = share(detail::create_with_size_from_view(
        exec, this->real_workspace_, offset, dim<2>{1, num_rhs}, b_stride));
    offset = b_stride;
    auto dense_tau = share(detail::create_with_size_from_view(
        exec, this->real_workspace_, offset, dim<2>{1, num_rhs}, b_stride));

    bool one_changed{};
    exec->run(ir::make_async_initialize(&this->stop_status_), handle);
    exec->run(ir::make_async_copy(detail::get_local(dense_b),
                                  detail::get_local(residual.get())),
              handle);

    this->get_system_matrix()->apply(lend(this->neg_one_op_), dense_x,
                                     lend(this->one_op_), lend(residual),
                                     handle);

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        lend(residual), this->one_op_, this->neg_one_op_, this->device_storage_,
        st_tau, dense_tau);

    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, lend(residual), dense_x);

        stop_criterion->update()
            .num_iterations(iter)
            .residual(lend(residual))
            .solution(dense_x)
            .check(handle, relative_stopping_id, true, &this->stop_status_,
                   &this->host_storage_)
            ->wait();

        if (this->host_storage_.get_data()[0]) {
            break;
        }

        // #if GINKGO_BUILD_MPI
        //         auto dist_mat =
        //             gko::as<const gko::distributed::Matrix<ValueType,
        //             int32>>(
        //                 this->get_system_matrix());

        //         if (solver_->apply_uses_initial_guess()) {
        //             // Use the inner solver to solve
        //             // A * inner_solution = residual
        //             // with residual as initial guess.
        //             inner_solution->copy_from(lend(residual));
        //             solver_->apply(lend(residual), lend(inner_solution));

        //             // x = x + relaxation_factor * inner_solution
        //             dense_x->add_scaled(lend(relaxation_factor_),
        //             lend(inner_solution));

        //             // residual = b - A * x
        //             residual->copy_from(dense_b);
        //             dist_mat->apply(lend(neg_one_op), dense_x, lend(one_op),
        //                             lend(residual));
        //         } else {
        //             // x = x + relaxation_factor * A \ residual
        //             solver_->apply(lend(relaxation_factor_), lend(residual),
        //                            lend(one_op), dense_x);

        //             // residual = b - A * x
        //             residual->copy_from(dense_b);
        //             dist_mat->apply(lend(neg_one_op), dense_x, lend(one_op),
        //                             lend(residual));
        //         }
        // #else
        if (this->get_solver()->apply_uses_initial_guess()) {
            // Use the inner solver to solve
            // A * inner_solution = residual
            // with residual as initial guess.
            // inner_solution->copy_from(lend(residual));
            exec->run(
                ir::make_async_copy(detail::get_local(residual.get()),
                                    detail::get_local(inner_solution.get())),
                handle);
            this->get_solver()->apply(lend(residual), lend(inner_solution),
                                      handle);

            // x = x + relaxation_factor * inner_solution
            dense_x->add_scaled(lend(relaxation_factor_), lend(inner_solution),
                                handle);

            // residual = b - A * x
            // residual->copy_from(dense_b);
            exec->run(ir::make_async_copy(detail::get_local(dense_b),
                                          detail::get_local(residual.get())),
                      handle);
            this->get_system_matrix()->apply(lend(this->neg_one_op_), dense_x,
                                             lend(this->one_op_),
                                             lend(residual), handle);
        } else {
            // x = x + relaxation_factor * A \ residual
            this->get_solver()->apply(lend(relaxation_factor_), lend(residual),
                                      lend(this->one_op_), dense_x, handle);

            // residual = b - A * x
            // residual->copy_from(dense_b);
            exec->run(ir::make_async_copy(detail::get_local(dense_b),
                                          detail::get_local(residual.get())),
                      handle);
            this->get_system_matrix()->apply(lend(this->neg_one_op_), dense_x,
                                             lend(this->one_op_),
                                             lend(residual), handle);
        }
        // #endif
    }
    return handle;
}


template <typename ValueType>
template <typename VectorType>
void Ir<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x,
                                     const OverlapMask& wmask) const
{
    using LocalVector = matrix::Dense<ValueType>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();
    auto one_op = initialize<LocalVector>({one<ValueType>()}, exec);
    auto neg_one_op = initialize<LocalVector>({-one<ValueType>()}, exec);

    // FIXME - Performance
    auto x_clone = as<VectorType>(dense_x->clone());
    auto residual = detail::create_with_same_size(dense_b);
    auto inner_solution = detail::create_with_same_size(dense_b);

    bool one_changed{};
    Array<stopping_status> stop_status(exec, dense_b->get_size()[1]);
    exec->run(ir::make_async_initialize(&stop_status),
              exec->get_default_exec_stream())
        ->wait();

    residual->copy_from(dense_b);
    this->get_system_matrix()->apply(lend(neg_one_op), x_clone.get(),
                                     lend(one_op), lend(residual));

    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}),
        x_clone.get(), lend(residual));

    int iter = -1;
    while (true) {
        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, lend(residual), x_clone.get());

        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(lend(residual))
                .solution(x_clone.get())
                .check(relative_stopping_id, true, &stop_status,
                       &one_changed)) {
            break;
        }

        if (solver_->apply_uses_initial_guess()) {
            // Use the inner solver to solve
            // A * inner_solution = residual
            // with residual as initial guess.
            inner_solution->copy_from(lend(residual));
            solver_->apply(lend(residual), lend(inner_solution));

            // x = x + relaxation_factor * inner_solution
            x_clone->add_scaled(lend(relaxation_factor_), lend(inner_solution));

            // residual = b - A * x
            residual->copy_from(dense_b);
            this->get_system_matrix()->apply(lend(neg_one_op), x_clone.get(),
                                             lend(one_op), lend(residual));
        } else {
            // x = x + relaxation_factor * A \ residual
            solver_->apply(lend(relaxation_factor_), lend(residual),
                           lend(one_op), x_clone.get());

            // residual = b - A * x
            residual->copy_from(dense_b);
            this->get_system_matrix()->apply(lend(neg_one_op), x_clone.get(),
                                             lend(one_op), lend(residual));
        }
    }
    // FIXME
    auto x_view = dense_x->create_submatrix(
        wmask.write_idxs, gko::span(0, dense_x->get_size()[1]));
    auto xclone_view = x_clone->create_submatrix(
        wmask.write_idxs, gko::span(0, dense_x->get_size()[1]));
    x_view->copy_from(xclone_view.get());
}


template <typename ValueType>
std::shared_ptr<AsyncHandle> Ir<ValueType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x,
    std::shared_ptr<AsyncHandle> handle) const
{
    return async_precision_dispatch_real_complex_distributed<ValueType>(
        [this, handle](auto dense_alpha, auto dense_b, auto dense_beta,
                       auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get(), handle);
            dense_x->scale(dense_beta, handle);
            dense_x->add_scaled(dense_alpha, x_clone.get(), handle);
            return handle;
        },
        alpha, b, beta, x);
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x) const
{
    auto handle = this->get_executor()->get_default_exec_stream();
    precision_dispatch_real_complex_distributed<ValueType>(
        [this, handle](auto dense_alpha, auto dense_b, auto dense_beta,
                       auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get(), handle)->wait();
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType>
void Ir<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x,
                               const OverlapMask& wmask) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this, wmask](auto dense_alpha, auto dense_b, auto dense_beta,
                      auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get(), wmask);
            auto x_view = dense_x->create_submatrix(
                wmask.write_idxs, span(0, dense_x->get_size()[1]));
            auto xclone_view = dense_x->create_submatrix(
                wmask.write_idxs, span(0, x_clone->get_size()[1]));
            x_view->scale(dense_beta);
            x_view->add_scaled(dense_alpha, xclone_view.get());
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_IR(_type) class Ir<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IR);


}  // namespace solver
}  // namespace gko
