// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/async_jacobi.hpp>
#include <ginkgo/core/solver/solver_base.hpp>

#include "core/solver/async_jacobi_kernels.hpp"
#include "core/solver/solver_boilerplate.hpp"


namespace gko {
namespace solver {
namespace async_jacobi {
namespace {


GKO_REGISTER_OPERATION(apply, async_jacobi::apply);


}  // anonymous namespace
}  // namespace async_jacobi


template <typename ValueType, typename IndexType>
void AsyncJacobi<ValueType, IndexType>::set_relaxation_factor(
    std::shared_ptr<const matrix::Dense<ValueType>> new_factor)
{
    auto exec = this->get_executor();
    if (new_factor && new_factor->get_executor() != exec) {
        new_factor = gko::clone(exec, new_factor);
    }
    relaxation_factor_ = new_factor;
}


template <typename ValueType, typename IndexType>
AsyncJacobi<ValueType, IndexType>& AsyncJacobi<ValueType, IndexType>::operator=(
    const AsyncJacobi& other)
{
    if (&other != this) {
        EnableLinOp<AsyncJacobi>::operator=(other);
        EnableSolverBase<AsyncJacobi>::operator=(other);
        EnableIterativeBase<AsyncJacobi>::operator=(other);
        this->set_relaxation_factor(other.relaxation_factor_);
        parameters_ = other.parameters_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
AsyncJacobi<ValueType, IndexType>& AsyncJacobi<ValueType, IndexType>::operator=(
    AsyncJacobi&& other)
{
    if (&other != this) {
        EnableLinOp<AsyncJacobi>::operator=(std::move(other));
        EnableSolverBase<AsyncJacobi>::operator=(std::move(other));
        EnableIterativeBase<AsyncJacobi>::operator=(std::move(other));
        this->set_relaxation_factor(other.relaxation_factor_);
        other.set_relaxation_factor(nullptr);
        parameters_ = other.parameters_;
    }
    return *this;
}


template <typename ValueType, typename IndexType>
AsyncJacobi<ValueType, IndexType>::AsyncJacobi(const AsyncJacobi& other)
    : AsyncJacobi(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
AsyncJacobi<ValueType, IndexType>::AsyncJacobi(AsyncJacobi&& other)
    : AsyncJacobi(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> AsyncJacobi<ValueType, IndexType>::transpose() const
{
    return build()
        .with_criteria(this->get_stop_criterion_factory())
        .with_relaxation_factor(parameters_.relaxation_factor)
        .on(this->get_executor())
        ->generate(
            share(as<Transposable>(this->get_system_matrix())->transpose()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> AsyncJacobi<ValueType, IndexType>::conj_transpose() const
{
    return build()
        .with_criteria(this->get_stop_criterion_factory())
        .with_relaxation_factor(conj(parameters_.relaxation_factor))
        .on(this->get_executor())
        ->generate(share(
            as<Transposable>(this->get_system_matrix())->conj_transpose()));
}


template <typename ValueType, typename IndexType>
void AsyncJacobi<ValueType, IndexType>::apply_impl(const LinOp* b,
                                                   LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            this->apply_dense_impl(dense_b, dense_x);
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void AsyncJacobi<ValueType, IndexType>::apply_dense_impl(
    const matrix::Dense<ValueType>* dense_b,
    matrix::Dense<ValueType>* dense_x) const
{
    using Vector = matrix::Dense<ValueType>;
    using ws = workspace_traits<AsyncJacobi>;
    using Csr = matrix::Csr<ValueType, IndexType>;
    constexpr uint8 relative_stopping_id{1};

    auto exec = this->get_executor();

    this->setup_workspace();

    GKO_SOLVER_VECTOR(residual, dense_b);
    GKO_SOLVER_VECTOR(inner_solution, dense_b);

    GKO_SOLVER_ONE_MINUS_ONE();
    exec->run(async_jacobi::make_apply(
        this->get_parameters().check, this->get_parameters().max_iters,
        relaxation_factor_.get(), second_factor_.get(),
        as<Csr>(this->get_system_matrix().get()), dense_b, dense_x));
}


template <typename ValueType, typename IndexType>
void AsyncJacobi<ValueType, IndexType>::apply_impl(const LinOp* alpha,
                                                   const LinOp* b,
                                                   const LinOp* beta,
                                                   LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto x_clone = dense_x->clone();
            this->apply_dense_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
int workspace_traits<AsyncJacobi<ValueType, IndexType>>::num_arrays(
    const Solver&)
{
    return 1;
}


template <typename ValueType, typename IndexType>
int workspace_traits<AsyncJacobi<ValueType, IndexType>>::num_vectors(
    const Solver&)
{
    return 4;
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<AsyncJacobi<ValueType, IndexType>>::op_names(const Solver&)
{
    return {
        "residual",
        "inner_solution",
        "one",
        "minus_one",
    };
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<AsyncJacobi<ValueType, IndexType>>::array_names(const Solver&)
{
    return {"stop"};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<AsyncJacobi<ValueType, IndexType>>::scalars(
    const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<AsyncJacobi<ValueType, IndexType>>::vectors(
    const Solver&)
{
    return {residual, inner_solution};
}


#define GKO_DECLARE_ASYNC_JACOBI(_vtype, _itype) \
    class AsyncJacobi<_vtype, _itype>
#define GKO_DECLARE_ASYNC_JACOBI_TRAITS(_vtype, _itype) \
    struct workspace_traits<AsyncJacobi<_vtype, _itype>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(GKO_DECLARE_ASYNC_JACOBI);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_ASYNC_JACOBI_TRAITS);


}  // namespace solver
}  // namespace gko
