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

#include <ginkgo/core/constraints/constrained_system.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/constraints/constrained_system_kernels.hpp"


namespace gko {
namespace constraints {
namespace cons {
namespace {


GKO_REGISTER_OPERATION(fill_subset, cons::fill_subset);
GKO_REGISTER_OPERATION(copy_subset, cons::copy_subset);
GKO_REGISTER_OPERATION(set_unit_rows, cons::set_unit_rows);


}  // namespace
}  // namespace cons


template <typename ValueType, typename IndexType>
std::unique_ptr<LinOp>
ZeroRowsStrategy<ValueType, IndexType>::construct_operator(
    const gko::Array<IndexType>& idxs, gko::LinOp* op)
{
    auto exec = op->get_executor();
    if (auto* csr = dynamic_cast<matrix::Csr<ValueType, IndexType>*>(op)) {
        exec->run(cons::make_set_unit_rows(idxs, csr->get_const_row_ptrs(),
                                           csr->get_const_col_idxs(),
                                           csr->get_values()));
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
        one = initialize<ValueType>({gko::one<ValueType>()}, exec);
    }
    if (!neg_one) {
        neg_one = initialize<ValueType>({-gko::one<ValueType>()}, exec);
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
    const gko::Array<gko::int32>& idxs, const gko::LinOp* op,
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
        one = initialize<ValueType>({gko::one<ValueType>()},
                                    solution->get_executor());
    }
    solution->add_scaled(one.get(), orig_init_guess);
    exec->run(cons::make_copy_subset(idxs, constrained_values->get_values(),
                                     solution->get_values()));
}

}  // namespace constraints
}  // namespace gko
