/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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


#include "core/solver/lower_trs_kernels.hpp"


namespace gko {
namespace solver {
namespace lower_trs {
namespace {


GKO_REGISTER_OPERATION(generate, lower_trs::generate);
GKO_REGISTER_OPERATION(should_perform_transpose,
                       lower_trs::should_perform_transpose);
GKO_REGISTER_OPERATION(solve, lower_trs::solve);


}  // anonymous namespace
}  // namespace lower_trs


template <typename ValueType, typename IndexType>
LowerTrs<ValueType, IndexType>::LowerTrs(const LowerTrs& other)
    : EnableLinOp<LowerTrs>(other.get_executor())
{
    *this = other;
}


template <typename ValueType, typename IndexType>
LowerTrs<ValueType, IndexType>::LowerTrs(LowerTrs&& other)
    : EnableLinOp<LowerTrs>(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType, typename IndexType>
LowerTrs<ValueType, IndexType>& LowerTrs<ValueType, IndexType>::operator=(
    const LowerTrs& other)
{
    if (this != &other) {
        EnableLinOp<LowerTrs>::operator=(other);
        EnableSolverBase<LowerTrs, CsrMatrix>::operator=(other);
        this->parameters_ = other.parameters_;
        this->generate();
    }
    return *this;
}


template <typename ValueType, typename IndexType>
LowerTrs<ValueType, IndexType>& LowerTrs<ValueType, IndexType>::operator=(
    LowerTrs&& other)
{
    if (this != &other) {
        EnableLinOp<LowerTrs>::operator=(std::move(other));
        EnableSolverBase<LowerTrs, CsrMatrix>::operator=(std::move(other));
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
std::unique_ptr<LinOp> LowerTrs<ValueType, IndexType>::transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->transpose()));
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp> LowerTrs<ValueType, IndexType>::conj_transpose() const
{
    return transposed_type::build()
        .with_num_rhs(this->parameters_.num_rhs)
        .on(this->get_executor())
        ->generate(share(this->get_system_matrix()->conj_transpose()));
}


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::generate()
{
    if (this->get_system_matrix()) {
        this->get_executor()->run(lower_trs::make_generate(
            this->get_system_matrix().get(), this->solve_struct_,
            this->get_parameters().unit_diagonal, parameters_.algorithm,
            parameters_.num_rhs));
    }
}


static bool needs_transpose(std::shared_ptr<const Executor> exec)
{
    bool result{};
    exec->run(lower_trs::make_should_perform_transpose(result));
    return result;
}


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::apply_impl(const LinOp* b, LinOp* x) const
{
    if (!this->get_system_matrix()) {
        return;
    }
    mixed_precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            using InputVector = matrix::Dense<
                typename std::decay_t<decltype(*dense_b)>::value_type>;
            using OutputVector = matrix::Dense<
                typename std::decay_t<decltype(*dense_x)>::value_type>;
            using ws = workspace_traits<LowerTrs>;
            const auto exec = this->get_executor();
            this->setup_workspace();

            // This kernel checks if a transpose is needed for the multiple rhs
            // case. Currently only the algorithm for HIP needs this
            // transposition due to the limitation in the hipsparse algorithm.
            // The other executors (omp and reference, CUDA) do not use the
            // transpose (trans_x and trans_b) and hence are passed in empty
            // pointers.
            InputVector* trans_b{};
            OutputVector* trans_x{};
            if (needs_transpose(exec)) {
                trans_b = this->template create_workspace_op<InputVector>(
                    ws::transposed_b, gko::transpose(dense_b->get_size()));
                trans_x = this->template create_workspace_op<OutputVector>(
                    ws::transposed_x, gko::transpose(dense_x->get_size()));
            }
            exec->run(lower_trs::make_solve(
                lend(this->get_system_matrix()), lend(this->solve_struct_),
                this->get_parameters().unit_diagonal, parameters_.algorithm,
                trans_b, trans_x, dense_b, dense_x));
        },
        b, x);
}


template <typename ValueType, typename IndexType>
void LowerTrs<ValueType, IndexType>::apply_impl(const LinOp* alpha,
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
            dense_x->add_scaled(dense_alpha, x_clone.get());
        },
        alpha, b, beta, x);
}


template <typename ValueType, typename IndexType>
int workspace_traits<LowerTrs<ValueType, IndexType>>::num_arrays(const Solver&)
{
    return 0;
}


template <typename ValueType, typename IndexType>
int workspace_traits<LowerTrs<ValueType, IndexType>>::num_vectors(
    const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? 2 : 0;
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<LowerTrs<ValueType, IndexType>>::op_names(const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? std::vector<std::string>{
        "transposed_b",
        "transposed_x",
    } : std::vector<std::string>{};
}


template <typename ValueType, typename IndexType>
std::vector<std::string>
workspace_traits<LowerTrs<ValueType, IndexType>>::array_names(const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<LowerTrs<ValueType, IndexType>>::scalars(
    const Solver&)
{
    return {};
}


template <typename ValueType, typename IndexType>
std::vector<int> workspace_traits<LowerTrs<ValueType, IndexType>>::vectors(
    const Solver& solver)
{
    return needs_transpose(solver.get_executor()) ? std::vector<int>{
        transposed_b,
        transposed_x,
    } : std::vector<int>{};
}


#define GKO_DECLARE_LOWER_TRS(_vtype, _itype) class LowerTrs<_vtype, _itype>
#define GKO_DECLARE_LOWER_TRS_TRAITS(_vtype, _itype) \
    struct workspace_traits<LowerTrs<_vtype, _itype>>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LOWER_TRS_TRAITS);


}  // namespace solver
}  // namespace gko
