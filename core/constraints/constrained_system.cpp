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


template <typename ValueType>
std::unique_ptr<LinOp> create_scalar(std::shared_ptr<const Executor> exec,
                                     ValueType v)
{
    return gko::initialize<matrix::Dense<ValueType>>({v}, exec);
}


template <template <typename, typename...> class ConcreteOp, typename Fn,
          typename... Args>
void value_type_dispatch(const LinOp* d, Fn&& fn, Args&&... args)
{
    if (dynamic_cast<const ConcreteOp<float>*>(d)) {
        fn(float{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const ConcreteOp<double>*>(d)) {
        fn(double{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const ConcreteOp<std::complex<float>>*>(d)) {
        fn(std::complex<float>{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const ConcreteOp<std::complex<double>>*>(d)) {
        fn(std::complex<double>{}, std::forward<Args>(args)...);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


template <template <typename, typename...> class ConcreteOp, typename Fn,
          typename... Args>
void index_type_dispatch(const LinOp* d, Fn&& fn, Args&&... args)
{
    if (dynamic_cast<const ConcreteOp<int32>*>(d)) {
        fn(int32{}, std::forward<Args>(args)...);
    } else if (dynamic_cast<const ConcreteOp<int64>*>(d)) {
        fn(int64{}, std::forward<Args>(args)...);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

struct empty {};
template <typename value_type = empty, typename index_type = empty>
struct type_match {
    type_match() = default;
    type_match(value_type, index_type) : found(true) {}

    operator bool() const { return found; }

    value_type vt = {};
    index_type it = {};
    bool found = false;
};


template <template <typename, typename, typename...> class ConcreteOp,
          typename Fn, typename... Args>
void value_index_type_dispatch(const LinOp* d, Fn&& fn, Args&&... args)
{
    auto check_index_type = [](auto vt, const auto* d) {
        using value_type = decltype(vt);
        if (dynamic_cast<const ConcreteOp<value_type, int32>*>(d)) {
            return type_match(vt, int32{});
        } else if (dynamic_cast<const ConcreteOp<value_type, int64>*>(d)) {
            return type_match(vt, int64{});
        } else {
            return type_match();
        }
    };
    if (auto match = check_index_type(float{}, d)) {
        fn(match.vt, match.it, std::forward<Args>(args)...);
    } else if (dynamic_cast<const ConcreteOp<double>*>(d)) {
        index_type_dispatch<ConcreteOp<float>>(
            d, [&](auto it) { fn(double{}, it, std::forward<Args>(args)...); });
    } else if (dynamic_cast<const ConcreteOp<std::complex<float>>*>(d)) {
        index_type_dispatch<ConcreteOp<float>>(d, [&](auto it) {
            fn(std::complex<float>{}, it, std::forward<Args>(args)...);
        });
    } else if (dynamic_cast<const ConcreteOp<std::complex<double>>*>(d)) {
        index_type_dispatch<ConcreteOp<float>>(d, [&](auto it) {
            fn(std::complex<double>{}, it, std::forward<Args>(args)...);
        });
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

std::unique_ptr<LinOp> create_one_with_same_value_type(const LinOp* v)
{
    auto exec = v->get_executor();
    auto one = v->create_default();
    value_type_dispatch<matrix::Dense>(
        v, [&](auto vt) { one = create_scalar(exec, gko::one(vt)); });
    return one;
}


std::unique_ptr<LinOp> create_neg_one_with_same_value_type(const LinOp* v)
{
    auto exec = v->get_executor();
    auto one = v->create_default();
    value_type_dispatch<matrix::Dense>(
        v, [&](auto vt) { one = create_scalar(exec, -gko::one(vt)); });
    return one;
}


std::unique_ptr<LinOp> ZeroRowsStrategy::construct_operator(
    const gko::Array<gko::int32>& idxs, gko::LinOp* op)
{
    value_index_type_dispatch<matrix::Csr>(op, [&](auto vt, auto it) {
        using value_type = decltype(vt);
        using index_type = decltype(it);
        auto csr = as<matrix::Csr<value_type, index_type>>(op);
    });
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
    value_type_dispatch<matrix::Dense>(cons_rhs.get(), [&](auto vt) {
        exec->run(cons::make_fill_subset(
            idxs,
            gko::as<matrix::Dense<decltype(vt)>>(cons_rhs.get())->get_values(),
            gko::zero(vt)));
    });
    return cons_rhs;
}


std::unique_ptr<LinOp> ZeroRowsStrategy::construct_initial_guess(
    const gko::Array<gko::int32>& idxs, const gko::LinOp* op,
    const LinOp* init_guess, const gko::LinOp* constrained_values)
{
    auto exec = init_guess->get_executor();
    auto cons_init_guess = gko::clone(init_guess);
    value_type_dispatch<matrix::Dense>(cons_init_guess.get(), [&](auto vt) {
        exec->run(cons::make_fill_subset(
            idxs,
            gko::as<matrix::Dense<decltype(vt)>>(cons_init_guess.get())
                ->get_values(),
            gko::zero(vt)));
    });
    return cons_init_guess;
}


void ZeroRowsStrategy::correct_solution(const gko::Array<gko::int32>& idxs,
                                        const gko::LinOp* orig_init_guess,
                                        gko::LinOp* solution)
{
    auto one = create_one_with_same_value_type(solution);
    value_type_dispatch<matrix::Dense>(solution, [&](auto vt) {
        using value_type = decltype(vt);
        auto* dense_solution = as<matrix::Dense<value_type>>(solution);
        dense_solution->add_scaled(one.get(), orig_init_guess);
    });
}

}  // namespace constraints
}  // namespace gko
