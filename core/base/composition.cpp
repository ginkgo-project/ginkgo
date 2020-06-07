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

#include <ginkgo/core/base/composition.hpp>


#include <algorithm>


#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array.hpp"


namespace gko {
namespace composition {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);


}  // namespace composition


template <typename ValueType>
std::unique_ptr<LinOp> apply_inner_operators(
    const std::vector<std::shared_ptr<const LinOp>> &operators,
    Array<ValueType> &storage, const LinOp *rhs)
{
    using Dense = matrix::Dense<ValueType>;
    // determine amount of necessary storage:
    // maximum sum of two subsequent intermediate vectors
    // (and the out dimension of the last op if we only have one operator)
    auto num_rhs = rhs->get_size()[1];
    auto max_intermediate_size = std::accumulate(
        begin(operators) + 1, end(operators) - 1,
        operators.back()->get_size()[0],
        [](size_type acc, std::shared_ptr<const LinOp> op) {
            return std::max(acc, op->get_size()[0] + op->get_size()[1]);
        });
    auto storage_size = max_intermediate_size * num_rhs;
    storage.resize_and_reset(storage_size);

    // apply inner vectors
    auto exec = rhs->get_executor();
    auto data = storage.get_data();
    // apply last operator
    auto out_dim = gko::dim<2>{operators.back()->get_size()[0], num_rhs};
    auto out = Dense::create(
        exec, out_dim, Array<ValueType>::view(exec, out_dim[0] * num_rhs, data),
        num_rhs);
    operators.back()->apply(rhs, lend(out));
    // apply following operators
    // alternate intermediate vectors between beginning/end of storage
    auto reversed_storage = true;
    for (auto i = operators.size() - 2; i > 0; --i) {
        // swap in and out
        auto in = std::move(out);
        // build new intermediate vector
        auto op_size = operators[i]->get_size();
        out_dim[0] = op_size[0];
        auto out_size = out_dim[0] * num_rhs;
        auto out_data =
            data + (reversed_storage ? storage_size - out_size : size_type{});
        reversed_storage = !reversed_storage;
        out = Dense::create(exec, out_dim,
                            Array<ValueType>::view(exec, out_size, out_data),
                            num_rhs);
        // for operators with initial guess: set initial guess
        if (operators[i]->apply_uses_initial_guess()) {
            if (op_size[0] == op_size[1]) {
                // square matrix: we can use the previous output
                exec->copy(out_size, in->get_const_values(), out->get_values());
            } else {
                // rectangular matrix: we can't do better than zeros
                exec->run(composition::make_fill_array(
                    out->get_values(), zero<ValueType>(), out_size));
            }
        }
        // apply operator
        operators[i]->apply(lend(in), lend(out));
    }

    return std::move(out);
}


template <typename ValueType>
void Composition<ValueType>::apply_impl(const LinOp *b, LinOp *x) const
{
    if (operators_.size() > 1) {
        operators_[0]->apply(
            lend(apply_inner_operators(operators_, storage_, b)), x);
    } else {
        operators_[0]->apply(b, x);
    }
}


template <typename ValueType>
void Composition<ValueType>::apply_impl(const LinOp *alpha, const LinOp *b,
                                        const LinOp *beta, LinOp *x) const
{
    if (operators_.size() > 1) {
        operators_[0]->apply(
            alpha, lend(apply_inner_operators(operators_, storage_, b)), beta,
            x);
    } else {
        operators_[0]->apply(alpha, b, beta, x);
    }
}


#define GKO_DECLARE_COMPOSITION(_type) class Composition<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMPOSITION);


}  // namespace gko
