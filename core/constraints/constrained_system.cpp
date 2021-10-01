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


template <typename ValueType>
std::unique_ptr<LinOp> create_scalar(std::shared_ptr<const Executor> exec,
                                     ValueType v)
{
    return gko::initialize<matrix::Dense<ValueType>>({v}, exec);
}


template <typename Fn, typename... Args>
void dense_value_type_dispatch(const LinOp* d, Fn&& fn, Args&&... args)
{
    if (dynamic_cast<const matrix::Dense<float>*>(d)) {
        fn(float{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const matrix::Dense<double>*>(d)) {
        fn(double{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const matrix::Dense<std::complex<float>>*>(d)) {
        fn(std::complex<float>{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const matrix::Dense<std::complex<double>>*>(d)) {
        fn(std::complex<double>{}, std::forward<Args>(args)...);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

std::unique_ptr<LinOp> create_one_with_same_value_type(const LinOp* v)
{
    auto exec = v->get_executor();
    auto one = v->create_default();
    dense_value_type_dispatch(
        v, [&](auto vt) { one = create_scalar(exec, gko::one(vt)); });
    return one;
}


std::unique_ptr<LinOp> create_neg_one_with_same_value_type(const LinOp* v)
{
    auto exec = v->get_executor();
    auto one = v->create_default();
    dense_value_type_dispatch(
        v, [&](auto vt) { one = create_scalar(exec, -gko::one(vt)); });
    return one;
}


std::unique_ptr<LinOp> ZeroRowsStrategy::construct_operator(
    const gko::Array<gko::int32>& idxs, gko::LinOp* op)
{
    return std::unique_ptr<LinOp>();
}


std::unique_ptr<LinOp> ZeroRowsStrategy::construct_right_hand_side(
    const gko::Array<gko::int32>& idxs, const gko::LinOp* op,
    const gko::LinOp* init_guess, const gko::LinOp* rhs)
{
    auto exec = rhs->get_executor();
    auto one = create_one_with_same_value_type(init_guess);
    auto neg_one = create_neg_one_with_same_value_type(init_guess);

    auto cons_rhs = gko::clone(rhs);
    op->apply(neg_one.get(), init_guess, one.get(), cons_rhs.get());
    dense_value_type_dispatch(cons_rhs.get(), [&](auto vt) {
        exec->run(cons::make_fill_subset(
            idxs,
            gko::as<matrix::Dense<decltype(vt)>>(cons_rhs.get())->get_values(),
            gko::zero(vt)));
    });
    return cons_rhs;
}


std::unique_ptr<LinOp> ZeroRowsStrategy::construct_initial_guess(
    const gko::Array<gko::int32>& idxs, const gko::LinOp* op,
    const gko::LinOp* constrained_values)
{
    return std::unique_ptr<LinOp>();
}


void ZeroRowsStrategy::correct_solution(const gko::Array<gko::int32>& idxs,
                                        const gko::LinOp* orig_init_guess,
                                        gko::LinOp* solution)
{}

}  // namespace constraints
}  // namespace gko
