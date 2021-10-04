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

#include <ginkgo/core/constraints/constraints_handler.hpp>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/constraints/constraints_handler_kernels.hpp"


namespace gko {
namespace constraints {
namespace cons {
namespace {


GKO_REGISTER_OPERATION(fill_subset, cons::fill_subset);
GKO_REGISTER_OPERATION(copy_subset, cons::copy_subset);
GKO_REGISTER_OPERATION(set_unit_rows, cons::set_unit_rows);


}  // namespace
}  // namespace cons


namespace detail {


template <typename ValueType, typename IndexType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
zero_guess_with_constrained_values(std::shared_ptr<const Executor> exec,
                                   dim<2> size, const Array<IndexType>& idxs,
                                   const matrix::Dense<ValueType>* values)
{
    using Dense = matrix::Dense<ValueType>;
    auto init = share(initialize<Dense>(0., size, exec));
    exec->run(cons::make_copy_subset(idxs, values->get_const_values(),
                                     init->get_values()));
    return init;
}


}  // namespace detail


template <typename ValueType, typename IndexType>
std::shared_ptr<LinOp>
ZeroRowsStrategy<ValueType, IndexType>::construct_operator(
    const Array<IndexType>& idxs, std::shared_ptr<LinOp> op)
{
    auto exec = op->get_executor();
    if (auto* csr =
            dynamic_cast<matrix::Csr<ValueType, IndexType>*>(op.get())) {
        exec->run(cons::make_set_unit_rows(idxs, csr->get_const_row_ptrs(),
                                           csr->get_const_col_idxs(),
                                           csr->get_values()));
        return op;
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp>
ZeroRowsStrategy<ValueType, IndexType>::construct_right_hand_side(
    const gko::Array<IndexType>& idxs, const gko::LinOp* op,
    const matrix::Dense<ValueType>* init_guess,
    const matrix::Dense<ValueType>* rhs)
{
    auto exec = rhs->get_executor();
    if (!one) {
        one = initialize<Dense>({gko::one<ValueType>()}, exec);
    }
    if (!neg_one) {
        neg_one = initialize<Dense>({-gko::one<ValueType>()}, exec);
    }

    auto cons_rhs = gko::clone(rhs);
    op->apply(neg_one.get(), init_guess, one.get(), cons_rhs.get());
    exec->run(cons::make_fill_subset(idxs, cons_rhs.get()->get_values(),
                                     gko::zero<ValueType>()));
    return cons_rhs;
}


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp>
ZeroRowsStrategy<ValueType, IndexType>::construct_initial_guess(
    const gko::Array<IndexType>& idxs, const gko::LinOp* op,
    const matrix::Dense<ValueType>* init_guess,
    const matrix::Dense<ValueType>* constrained_values)
{
    auto exec = init_guess->get_executor();
    auto cons_init_guess = gko::clone(init_guess);
    exec->run(cons::make_fill_subset(idxs, cons_init_guess.get()->get_values(),
                                     gko::zero<ValueType>()));
    return cons_init_guess;
}


template <typename ValueType, typename IndexType>
void ZeroRowsStrategy<ValueType, IndexType>::correct_solution(
    const gko::Array<IndexType>& idxs,
    const matrix::Dense<ValueType>* constrained_values,
    const matrix::Dense<ValueType>* orig_init_guess,
    matrix::Dense<ValueType>* solution)
{
    auto exec = solution->get_executor();
    if (!one) {
        one = initialize<Dense>({gko::one<ValueType>()},
                                solution->get_executor());
    }
    solution->add_scaled(one.get(), orig_init_guess);
    exec->run(cons::make_copy_subset(
        idxs, constrained_values->get_const_values(), solution->get_values()));
}


#define GKO_DECLARE_ZERO_ROWS_STRATEGY(ValueType, IndexType) \
    class ZeroRowsStrategy<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ZERO_ROWS_STRATEGY);


template <typename ValueType, typename IndexType>
ConstraintsHandler<ValueType, IndexType>::ConstraintsHandler(
    Array<IndexType> idxs, std::shared_ptr<LinOp> system_operator,
    std::shared_ptr<const Dense> values,
    std::shared_ptr<const Dense> right_hand_side,
    std::shared_ptr<const Dense> initial_guess,
    std::unique_ptr<ApplyConstraintsStrategy<ValueType, IndexType>> strategy)
    : idxs_(std::move(idxs)),
      orig_operator_(std::move(system_operator)),
      cons_operator_(strategy->construct_operator(idxs_, orig_operator_)),
      strategy_(std::move(strategy))
{
    // the order of the with_* calls matters.
    // with_initial_guess depends on the rhs
    // with_constrained_values depends on the initial guess
    this->with_right_hand_side(std::move(right_hand_side));
    if (initial_guess) {
        this->with_initial_guess(std::move(initial_guess));
    }
    this->with_constrained_values(std::move(values));
    reconstruct_system();
}


template <typename ValueType, typename IndexType>
ConstraintsHandler<ValueType, IndexType>::ConstraintsHandler(
    Array<IndexType> idxs, std::shared_ptr<LinOp> system_operator,
    std::unique_ptr<ApplyConstraintsStrategy<ValueType, IndexType>> strategy)
    : idxs_(std::move(idxs)),
      orig_operator_(std::move(system_operator)),
      cons_operator_(strategy->construct_operator(idxs_, orig_operator_)),
      strategy_(std::move(strategy))
{}


template <typename ValueType, typename IndexType>
ConstraintsHandler<ValueType, IndexType>&
ConstraintsHandler<ValueType, IndexType>::with_constrained_values(
    std::shared_ptr<const Dense> values)
{
    values_ = std::move(values);

    if (!cons_init_guess_) {
        auto exec = orig_rhs_ ? orig_rhs_->get_executor()
                              : orig_operator_->get_executor();
        auto size = orig_rhs_ ? orig_rhs_->get_size()
                              : dim<2>{orig_operator_->get_size()[0], 1};
        zero_init_guess_ = detail::zero_guess_with_constrained_values(
            exec, size, idxs_, values_.get());
    }

    // invalidate previous pointers
    cons_init_guess_.reset();
    cons_rhs_.reset();

    return *this;
}


template <typename ValueType, typename IndexType>
ConstraintsHandler<ValueType, IndexType>&
ConstraintsHandler<ValueType, IndexType>::with_right_hand_side(
    std::shared_ptr<const Dense> right_hand_side)
{
    orig_rhs_ = std::move(right_hand_side);

    // invalidate previous pointer
    cons_rhs_.reset();

    return *this;
}


template <typename ValueType, typename IndexType>
ConstraintsHandler<ValueType, IndexType>&
ConstraintsHandler<ValueType, IndexType>::with_initial_guess(
    std::shared_ptr<const Dense> initial_guess)
{
    orig_init_guess_ = std::move(initial_guess);

    // invalidate previous pointers
    cons_init_guess_.reset();
    cons_rhs_.reset();

    return *this;
}


template <typename ValueType, typename IndexType>
std::shared_ptr<const LinOp>
ConstraintsHandler<ValueType, IndexType>::get_operator()
{
    return cons_operator_;
}


template <typename ValueType, typename IndexType>
const LinOp* ConstraintsHandler<ValueType, IndexType>::get_right_hand_side()
{
    if (!cons_rhs_) {
        if (!cons_init_guess_) {
            reconstruct_system();
        } else {
            cons_rhs_ = as<Dense>(strategy_->construct_right_hand_side(
                idxs_, lend(cons_operator_), lend(used_init_guess()),
                lend(orig_rhs_)));
        }
    }
    return cons_rhs_.get();
}


template <typename ValueType, typename IndexType>
LinOp* ConstraintsHandler<ValueType, IndexType>::get_initial_guess()
{
    if (!cons_init_guess_) {
        cons_init_guess_ = as<Dense>(strategy_->construct_initial_guess(
            idxs_, lend(cons_operator_), lend(used_init_guess()),
            lend(values_)));
    }
    return cons_init_guess_.get();
}


template <typename ValueType, typename IndexType>
void ConstraintsHandler<ValueType, IndexType>::reconstruct_system()
{
    cons_init_guess_ = as<Dense>(strategy_->construct_initial_guess(
        idxs_, lend(cons_operator_), lend(used_init_guess()), lend(values_)));
    cons_rhs_ = as<Dense>(strategy_->construct_right_hand_side(
        idxs_, lend(cons_operator_), lend(used_init_guess()), lend(orig_rhs_)));
}


template <typename ValueType, typename IndexType>
void ConstraintsHandler<ValueType, IndexType>::correct_solution(Dense* solution)
{
    strategy_->correct_solution(idxs_, lend(values_), lend(used_init_guess()),
                                solution);
}


#define GKO_DECLARE_CONSTRAINTS_HANDLER(ValueType, IndexType) \
    class ConstraintsHandler<ValueType, IndexType>
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONSTRAINTS_HANDLER);


}  // namespace constraints
}  // namespace gko
