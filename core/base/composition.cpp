// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/composition.hpp>


#include <algorithm>
#include <iterator>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array_kernels.hpp"


namespace gko {
namespace composition {
namespace {


GKO_REGISTER_OPERATION(fill_array, components::fill_array);


}  // anonymous namespace
}  // namespace composition


template <typename ValueType>
std::unique_ptr<LinOp> apply_inner_operators(
    const std::vector<std::shared_ptr<const LinOp>>& operators,
    array<ValueType>& storage, const LinOp* rhs)
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
    auto op_size = operators.back()->get_size();
    auto out_dim = gko::dim<2>{op_size[0], num_rhs};
    auto out_size = out_dim[0] * num_rhs;
    auto out = Dense::create(exec, out_dim,
                             make_array_view(exec, out_size, data), num_rhs);
    // for operators with initial guess: set initial guess
    if (operators.back()->apply_uses_initial_guess()) {
        if (op_size[0] == op_size[1]) {
            // square matrix: we can use the previous output
            exec->copy(out_size, as<Dense>(rhs)->get_const_values(),
                       out->get_values());
        } else {
            // rectangular matrix: we can't do better than zeros
            exec->run(composition::make_fill_array(out->get_values(), out_size,
                                                   zero<ValueType>()));
        }
    }
    operators.back()->apply(rhs, out);
    // apply following operators
    // alternate intermediate vectors between beginning/end of storage
    auto reversed_storage = true;
    for (auto i = operators.size() - 2; i > 0; --i) {
        // swap in and out
        auto in = std::move(out);
        // build new intermediate vector
        op_size = operators[i]->get_size();
        out_dim[0] = op_size[0];
        out_size = out_dim[0] * num_rhs;
        auto out_data =
            data + (reversed_storage ? storage_size - out_size : size_type{});
        reversed_storage = !reversed_storage;
        out = Dense::create(exec, out_dim,
                            make_array_view(exec, out_size, out_data), num_rhs);
        // for operators with initial guess: set initial guess
        if (operators[i]->apply_uses_initial_guess()) {
            if (op_size[0] == op_size[1]) {
                // square matrix: we can use the previous output
                exec->copy(out_size, in->get_const_values(), out->get_values());
            } else {
                // rectangular matrix: we can't do better than zeros
                exec->run(composition::make_fill_array(
                    out->get_values(), out_size, zero<ValueType>()));
            }
        }
        // apply operator
        operators[i]->apply(in, out);
    }

    return std::move(out);
}


template <typename ValueType>
Composition<ValueType>& Composition<ValueType>::operator=(
    const Composition& other)
{
    if (&other != this) {
        EnableLinOp<Composition>::operator=(other);
        auto exec = this->get_executor();
        operators_ = other.operators_;
        // if the operators are on the wrong executor, copy them over
        if (other.get_executor() != exec) {
            for (auto& op : operators_) {
                op = gko::clone(exec, op);
            }
        }
    }
    return *this;
}


template <typename ValueType>
Composition<ValueType>& Composition<ValueType>::operator=(Composition&& other)
{
    if (&other != this) {
        EnableLinOp<Composition>::operator=(std::move(other));
        auto exec = this->get_executor();
        operators_ = std::move(other.operators_);
        // if the operators are on the wrong executor, copy them over
        if (other.get_executor() != exec) {
            for (auto& op : operators_) {
                op = gko::clone(exec, op);
            }
        }
    }
    return *this;
}


template <typename ValueType>
Composition<ValueType>::Composition(const Composition& other)
    : Composition(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Composition<ValueType>::Composition(Composition&& other)
    : Composition(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
std::unique_ptr<LinOp> Composition<ValueType>::transpose() const
{
    auto transposed = Composition<ValueType>::create(this->get_executor());
    transposed->set_size(gko::transpose(this->get_size()));
    // transpose and reverse operators
    std::transform(this->get_operators().rbegin(), this->get_operators().rend(),
                   std::back_inserter(transposed->operators_),
                   [](const std::shared_ptr<const LinOp>& op) {
                       return share(as<Transposable>(op)->transpose());
                   });

    return std::move(transposed);
}


template <typename ValueType>
std::unique_ptr<LinOp> Composition<ValueType>::conj_transpose() const
{
    auto transposed = Composition<ValueType>::create(this->get_executor());
    transposed->set_size(gko::transpose(this->get_size()));
    // conjugate-transpose and reverse operators
    std::transform(this->get_operators().rbegin(), this->get_operators().rend(),
                   std::back_inserter(transposed->operators_),
                   [](const std::shared_ptr<const LinOp>& op) {
                       return share(as<Transposable>(op)->conj_transpose());
                   });

    return std::move(transposed);
}


template <typename ValueType>
void Composition<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            if (operators_.size() > 1) {
                operators_[0]->apply(
                    apply_inner_operators(operators_, storage_, dense_b),
                    dense_x);
            } else {
                operators_[0]->apply(dense_b, dense_x);
            }
        },
        b, x);
}


template <typename ValueType>
void Composition<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                        const LinOp* beta, LinOp* x) const
{
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            if (operators_.size() > 1) {
                operators_[0]->apply(
                    dense_alpha,
                    apply_inner_operators(operators_, storage_, dense_b),
                    dense_beta, dense_x);
            } else {
                operators_[0]->apply(dense_alpha, dense_b, dense_beta, dense_x);
            }
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_COMPOSITION(_type) class Composition<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMPOSITION);


}  // namespace gko
