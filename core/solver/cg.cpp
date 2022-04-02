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

#include <ginkgo/core/solver/cg.hpp>


#include <ginkgo/core/base/async_handle.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/memory_space.hpp>
#include <ginkgo/core/base/name_demangling.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>


#include "core/base/handle_guard.hpp"
#include "core/distributed/helpers.hpp"
#include "core/solver/cg_kernels.hpp"


namespace gko {
namespace solver {
namespace cg {
namespace {


GKO_REGISTER_ASYNC_OPERATION(initialize, cg::initialize);
GKO_REGISTER_ASYNC_OPERATION(step_1, cg::step_1);
GKO_REGISTER_ASYNC_OPERATION(step_2, cg::step_2);


}  // anonymous namespace
}  // namespace cg


template <typename ValueType>
Cg<ValueType>& Cg<ValueType>::operator=(const Cg& other)
{
    if (&other != this) {
        EnableLinOp<Cg>::operator=(other);
        EnableSolverBase<Cg>::operator=(other);
        EnableIterativeBase<Cg>::operator=(other);
        parameters_ = other.parameters_;
        if (this->get_system_matrix()) {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType>
Cg<ValueType>& Cg<ValueType>::operator=(Cg&& other)
{
    if (&other != this) {
        EnableLinOp<Cg>::operator=(std::move(other));
        EnableSolverBase<Cg>::operator=(std::move(other));
        EnableIterativeBase<Cg>::operator=(std::move(other));
        parameters_ = other.parameters_;
        if (this->get_system_matrix()) {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType>
Cg<ValueType>::Cg(const Cg& other) : Cg(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Cg<ValueType>::Cg(Cg&& other) : Cg(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
void Cg<ValueType>::generate()
{
    // TODO: Fix for distributed. Need to have local sizes
    auto num_rows = this->get_system_matrix()->get_size()[0];
    // FIXME: Probably needs to be stride instead
    auto nrhs = this->get_parameters().num_rhs;
    if (this->get_executor()) {
        this->workspace_ = gko::Array<ValueType>(
            this->get_executor(),
            num_rows * nrhs * num_aux_vecs + nrhs * num_aux_scalars);
        this->real_workspace_ = gko::Array<remove_complex<ValueType>>(
            this->get_executor(), nrhs * 2);
        this->stop_status_ =
            gko::Array<stopping_status>(this->get_executor(), nrhs);
        this->device_storage_ =
            std::make_shared<Array<bool>>(this->get_executor(), 2);
        this->one_op_ = initialize<matrix::Dense<ValueType>>(
            {one<ValueType>()}, this->get_executor());
        this->neg_one_op_ = initialize<matrix::Dense<ValueType>>(
            {-one<ValueType>()}, this->get_executor());
        this->host_storage_ =
            gko::Array<bool>(this->get_executor()->get_master(), 2,
                             this->get_executor()
                                 ->get_mem_space()
                                 ->template pinned_host_alloc<bool>(2),
                             memory_space_pinned_host_deleter<bool[]>(
                                 this->get_executor()->get_mem_space()));
        this->host_storage_.fill(false);
    }
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
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType>
std::shared_ptr<AsyncHandle> Cg<ValueType>::apply_impl(
    const LinOp* b, LinOp* x, std::shared_ptr<AsyncHandle> handle) const
{
    return async_precision_dispatch_real_complex_distributed<ValueType>(
        [this, handle](auto dense_b, auto dense_x) {
            return this->apply_dense_impl(dense_b, dense_x, handle);
        },
        b, x);
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp* b, LinOp* x,
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
std::shared_ptr<AsyncHandle> Cg<ValueType>::apply_dense_impl(
    const VectorType* dense_b, VectorType* dense_x,
    std::shared_ptr<AsyncHandle> handle) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();
    handle_guard hg{exec, handle};

    auto num_rhs = dense_b->get_size()[1];
    auto num_rows = detail::get_local(dense_b)->get_size()[0];
    auto b_stride = detail::get_local(dense_b)->get_stride();
    if (this->get_parameters().num_rhs < b_stride) {
        this->workspace_ = gko::Array<ValueType>(
            this->get_executor(),
            num_rows * b_stride * num_aux_vecs + b_stride * num_aux_scalars);
        this->real_workspace_ = gko::Array<remove_complex<ValueType>>(
            this->get_executor(), b_stride * 2);
        this->stop_status_ =
            gko::Array<stopping_status>(this->get_executor(), b_stride);
    }
    int offset = 0;
    auto r = detail::create_with_same_size_from_view(this->workspace_, offset,
                                                     dense_b);
    offset += num_rows * b_stride;
    auto z = detail::create_with_same_size_from_view(this->workspace_, offset,
                                                     dense_b);
    offset += num_rows * b_stride;
    auto p = detail::create_with_same_size_from_view(this->workspace_, offset,
                                                     dense_b);
    offset += num_rows * b_stride;
    auto q = detail::create_with_same_size_from_view(this->workspace_, offset,
                                                     dense_b);

    offset += num_rows * b_stride;
    auto alpha = detail::create_with_size_from_view(exec, this->workspace_,
                                                    offset, dim<2>{1, num_rhs});
    offset += num_rhs;
    auto beta = detail::create_with_same_size_from_view(this->workspace_,
                                                        offset, alpha.get());
    offset += num_rhs;
    auto prev_rho = detail::create_with_same_size_from_view(
        this->workspace_, offset, alpha.get());
    offset += num_rhs;
    auto rho = detail::create_with_same_size_from_view(this->workspace_, offset,
                                                       alpha.get());
    offset = 0;
    auto st_tau = share(detail::create_with_size_from_view(
        exec, this->real_workspace_, offset, dim<2>{1, num_rhs}, b_stride));
    offset = b_stride;
    auto dense_tau = share(detail::create_with_size_from_view(
        exec, this->real_workspace_, offset, dim<2>{1, num_rhs}, b_stride));

    bool one_changed{};

    // TODO: replace this with automatic merged kernel generator
    exec->run(cg::make_async_initialize(
                  detail::get_local(dense_b), detail::get_local(r.get()),
                  detail::get_local(z.get()), detail::get_local(p.get()),
                  detail::get_local(q.get()), prev_rho.get(), rho.get(),
                  &this->stop_status_),
              handle);
    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0

    this->get_system_matrix()->apply(this->neg_one_op_.get(), dense_x,
                                     this->one_op_.get(), r.get(), handle);
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get(), this->one_op_, this->neg_one_op_, this->device_storage_,
        st_tau, dense_tau);


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
        this->get_preconditioner()->apply(r.get(), z.get(), handle);
        r->compute_conj_dot(z.get(), rho.get(), handle);

        ++iter;
        // this->template log<log::Logger::iteration_complete>(
        //     this, iter, r.get(), dense_x, nullptr, rho.get());
        auto stop = stop_criterion->update()
                        .num_iterations(iter)
                        .residual(r.get())
                        .implicit_sq_residual_norm(rho.get())
                        .solution(dense_x)
                        .check(handle, RelativeStoppingId, true,
                               &this->stop_status_, &this->host_storage_);

        if (this->host_storage_.get_data()[0]) {
            break;
        }

        // tmp = rho / prev_rho
        // p = z + tmp * p
        exec->run(cg::make_async_step_1(detail::get_local(p.get()),
                                        detail::get_local(z.get()), rho.get(),
                                        prev_rho.get(), &this->stop_status_),
                  handle);

        this->get_system_matrix()->apply(p.get(), q.get(), handle);
        p->compute_conj_dot(q.get(), beta.get(), handle);
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(cg::make_async_step_2(
                      detail::get_local(dense_x), detail::get_local(r.get()),
                      detail::get_local(p.get()), detail::get_local(q.get()),
                      beta.get(), rho.get(), &this->stop_status_),
                  handle);
        // handle->wait();
        swap(prev_rho, rho);
    }
    return handle;
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

    auto r = detail::create_with_same_size(dense_b);
    auto z = detail::create_with_same_size(dense_b);
    auto p = detail::create_with_same_size(dense_b);
    auto q = detail::create_with_same_size(dense_b);

    auto alpha = LocalVector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = LocalVector::create_with_config_of(alpha.get());
    auto prev_rho = LocalVector::create_with_config_of(alpha.get());
    auto rho = LocalVector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(cg::make_async_initialize(
                  detail::get_local(dense_b), detail::get_local(r.get()),
                  detail::get_local(z.get()), detail::get_local(p.get()),
                  detail::get_local(q.get()), prev_rho.get(), rho.get(),
                  &stop_status),
              exec->get_default_exec_stream())
        ->wait();
    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0

    GKO_ASSERT(dense_x->get_executor() != nullptr);
    GKO_ASSERT(r->get_executor() != nullptr);
    GKO_ASSERT(this->neg_one_op_->get_executor() != nullptr);
    GKO_ASSERT(this->one_op_->get_executor() != nullptr);
    GKO_ASSERT(this->get_system_matrix()->get_executor() != nullptr);
    this->get_system_matrix()->apply(this->neg_one_op_.get(), dense_x,
                                     this->one_op_.get(), r.get());
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}), dense_x,
        r.get());


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
        this->get_preconditioner()->apply(r.get(), z.get());
        r->compute_conj_dot(z.get(), rho.get());

        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r.get(), dense_x, nullptr, rho.get());
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .implicit_sq_residual_norm(rho.get())
                .solution(dense_x)
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho / prev_rho
        // p = z + tmp * p
        exec->run(cg::make_async_step_1(detail::get_local(p.get()),
                                        detail::get_local(z.get()), rho.get(),
                                        prev_rho.get(), &stop_status),
                  exec->get_default_exec_stream())
            ->wait();
        this->get_system_matrix()->apply(p.get(), q.get());
        p->compute_conj_dot(q.get(), beta.get());
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(cg::make_async_step_2(
                      detail::get_local(dense_x), detail::get_local(r.get()),
                      detail::get_local(p.get()), detail::get_local(q.get()),
                      beta.get(), rho.get(), &stop_status),
                  exec->get_default_exec_stream())
            ->wait();
        swap(prev_rho, rho);
    }
}


template <typename ValueType>
template <typename VectorType>
void Cg<ValueType>::apply_dense_impl(const VectorType* dense_b,
                                     VectorType* dense_x,
                                     const OverlapMask& wmask) const
{
    using std::swap;
    using LocalVector = matrix::Dense<ValueType>;

    constexpr uint8 RelativeStoppingId{1};

    auto exec = this->get_executor();

    auto r = detail::create_with_same_size(dense_b);
    auto z = detail::create_with_same_size(dense_b);
    auto p = detail::create_with_same_size(dense_b);
    auto q = detail::create_with_same_size(dense_b);

    auto alpha = LocalVector::create(exec, dim<2>{1, dense_b->get_size()[1]});
    auto beta = LocalVector::create_with_config_of(alpha.get());
    auto prev_rho = LocalVector::create_with_config_of(alpha.get());
    auto rho = LocalVector::create_with_config_of(alpha.get());

    bool one_changed{};
    Array<stopping_status> stop_status(alpha->get_executor(),
                                       dense_b->get_size()[1]);

    // TODO: replace this with automatic merged kernel generator
    exec->run(cg::make_async_initialize(
                  detail::get_local(dense_b), detail::get_local(r.get()),
                  detail::get_local(z.get()), detail::get_local(p.get()),
                  detail::get_local(q.get()), prev_rho.get(), rho.get(),
                  &stop_status),
              exec->get_default_exec_stream())
        ->wait();
    // r = dense_b
    // rho = 0.0
    // prev_rho = 1.0
    // z = p = q = 0

    GKO_ASSERT(dense_x->get_executor() != nullptr);
    GKO_ASSERT(r->get_executor() != nullptr);
    GKO_ASSERT(this->neg_one_op_->get_executor() != nullptr);
    GKO_ASSERT(this->one_op_->get_executor() != nullptr);
    GKO_ASSERT(this->get_system_matrix()->get_executor() != nullptr);
    auto x_clone = as<VectorType>(dense_x->clone());
    this->get_system_matrix()->apply(this->neg_one_op_.get(), x_clone.get(),
                                     this->one_op_.get(), r.get());
    auto stop_criterion = this->get_stop_criterion_factory()->generate(
        this->get_system_matrix(),
        std::shared_ptr<const LinOp>(dense_b, [](const LinOp*) {}),
        x_clone.get(), r.get());

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
        this->get_preconditioner()->apply(r.get(), z.get());
        r->compute_conj_dot(z.get(), rho.get());

        ++iter;
        this->template log<log::Logger::iteration_complete>(
            this, iter, r.get(), dense_x, nullptr, rho.get());
        if (stop_criterion->update()
                .num_iterations(iter)
                .residual(r.get())
                .implicit_sq_residual_norm(rho.get())
                .solution(x_clone.get())
                .check(RelativeStoppingId, true, &stop_status, &one_changed)) {
            break;
        }

        // tmp = rho / prev_rho
        // p = z + tmp * p
        exec->run(cg::make_async_step_1(detail::get_local(p.get()),
                                        detail::get_local(z.get()), rho.get(),
                                        prev_rho.get(), &stop_status),
                  exec->get_default_exec_stream())
            ->wait();
        this->get_system_matrix()->apply(p.get(), q.get());
        p->compute_conj_dot(q.get(), beta.get());
        // tmp = rho / beta
        // x = x + tmp * p
        // r = r - tmp * q
        exec->run(cg::make_async_step_2(detail::get_local(x_clone.get()),
                                        detail::get_local(r.get()),
                                        detail::get_local(p.get()),
                                        detail::get_local(q.get()), beta.get(),
                                        rho.get(), &stop_status),
                  exec->get_default_exec_stream())
            ->wait();
        swap(prev_rho, rho);
    }
    // FIXME
    auto x_view = dense_x->create_submatrix(
        wmask.write_idxs, gko::span(0, dense_x->get_size()[1]));
    auto xclone_view = x_clone->create_submatrix(
        wmask.write_idxs, gko::span(0, dense_x->get_size()[1]));
    x_view->copy_from(xclone_view.get());
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                               const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType>
std::shared_ptr<AsyncHandle> Cg<ValueType>::apply_impl(
    const LinOp* alpha, const LinOp* b, const LinOp* beta, LinOp* x,
    std::shared_ptr<AsyncHandle> handle) const
{
    return async_precision_dispatch_real_complex_distributed<ValueType>(
        [this, handle](auto dense_alpha, auto dense_b, auto dense_beta,
                       auto dense_x) {
            handle_guard hg{dense_x->get_executor(), handle};
            auto x_clone = dense_x->clone();
            auto hand1 = this->apply_dense_impl(dense_b, x_clone.get(), handle);
            auto hand2 = dense_x->scale(dense_beta, hand1);
            return dense_x->add_scaled(dense_alpha, x_clone.get(), hand2);
        },
        alpha, b, beta, x);
}


template <typename ValueType>
void Cg<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
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


#define GKO_DECLARE_CG(_type) class Cg<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CG);


}  // namespace solver
}  // namespace gko
