// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/polymorphic_object.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/triangular.hpp>


#include "core/solver/upper_trs_kernels.hpp"


namespace gko {
namespace solver {
namespace upper_trs {
namespace {


GKO_REGISTER_OPERATION(generate, upper_trs::generate);
GKO_REGISTER_OPERATION(should_perform_transpose,
                       upper_trs::should_perform_transpose);
GKO_REGISTER_OPERATION(solve, upper_trs::solve);


}  // anonymous namespace
}  // namespace upper_trs


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(const UpperTrs& other)
    : EnableLinOp<UpperTrs>(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>::UpperTrs(UpperTrs&& other)
    : EnableLinOp<UpperTrs>(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>& UpperTrs<ValueType, IndexType>::operator=(
    const UpperTrs& other)
{
    if (this != &other) {
        EnableLinOp<UpperTrs>::operator=(other);
        EnableSolverBase<UpperTrs, CsrMatrix>::operator=(other);
        this->parameters_ = other.parameters_;
        this->generate();
    }
    return *this;
}


template <typename ValueType, typename IndexType>
UpperTrs<ValueType, IndexType>& UpperTrs<ValueType, IndexType>::operator=(
    UpperTrs&& other)
{
    if (this != &other) {
        EnableLinOp<UpperTrs>::operator=(std::move(other));
        EnableSolverBase<UpperTrs, CsrMatrix>::operator=(std::move(other));
        this->parameters_ = std::exchange(other.parameters_, parameters_type{});
        if (this->get_executor() == other.get_executor()) {
            this->solve_struct_ = std::exchange(other.solve_struct_, nullptr);
        } else {
            this->generate();
        }
    }
    return *this;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> UpperTrs<ValueType, IndexType>::transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->transpose()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> UpperTrs<ValueType, IndexType>::conj_transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->conj_transpose()));
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::generate()
{
    if (this->get_system_matrix()) {
        this->get_executor()->run(upper_trs::make_generate(
            this->get_system_matrix().get(), this->solve_struct_,
            this->get_parameters().unit_diagonal, parameters_.algorithm,
            parameters_.num_rhs));
    }
}


static bool needs_transpose(std::shared_ptr<const Executor> exec)
{
    bool result{};
    exec->run(upper_trs::make_should_perform_transpose(result));
    return result;
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            using Vector = matrix::Dense<ValueType>;
            using ws = workspace_traits<UpperTrs>;
            const auto exec = this->get_executor();
            this->setup_workspace();

            // This kernel checks if a transpose is needed for the multiple rhs
            // case. Currently only the algorithm for HIP needs this
            // transposition due to the limitation in the hipsparse algorithm.
            // The other executors (omp and reference, CUDA) do not use the
            // transpose (trans_x and trans_b) and hence are passed in empty
            // pointers.
            Vector* trans_b{};
            Vector* trans_x{};
            if (needs_transpose(exec)) {
                trans_b = this->template create_workspace_op<Vector>(
                    ws::transposed_b, gko::transpose(dense_b->get_size()));
                trans_x = this->template create_workspace_op<Vector>(
                    ws::transposed_x, gko::transpose(dense_x->get_size()));
            }
            exec->run(upper_trs::make_solve(
                this->get_system_matrix().get(), this->solve_struct_.get(),
                this->get_parameters().unit_diagonal, parameters_.algorithm,
                trans_b, trans_x, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void UpperTrs<ValueType, IndexType>::apply_impl(const LinOp* alpha,
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
            this->apply_impl(dense_b, x_clone.get());
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, x_clone);
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
int workspace_traits<UpperTrs<ValueType, IndexType>>::num_arrays(const Solver&)
{
    return 0;
}


template <typename ValueType, typename IndexType>
int workspace_traits<UpperTrs<ValueType, IndexType>>::num_vectors(
    const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? 2 : 0;
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<UpperTrs<ValueType, IndexType>>::op_names(const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? std::vector<std::string>{
        "transposed_b",
        "transposed_x",
    } : std::vector<std::string>{};
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<UpperTrs<ValueType, IndexType>>::array_names(const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<UpperTrs<ValueType, IndexType>>::scalars(
    const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<UpperTrs<ValueType, IndexType>>::vectors(
    const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? std::vector<int>{
        transposed_b,
        transposed_x,
    } : std::vector<int>{};
}


#define GKO_DECLARE_UPPER_TRS(_vtype, _itype) class UpperTrs<_vtype, _itype>
#define GKO_DECLARE_UPPER_TRS_TRAITS(_vtype, _itype) \
    struct workspace_traits<UpperTrs<_vtype, _itype>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_UPPER_TRS_TRAITS);


}  // namespace solver
}  // namespace gko
