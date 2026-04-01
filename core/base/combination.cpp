// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/combination.hpp"

#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace {


template <typename ValueType>
inline void initialize_scalars(std::shared_ptr<const Executor> exec,
                               std::unique_ptr<matrix::Dense<ValueType>>& zero,
                               std::unique_ptr<matrix::Dense<ValueType>>& one)
{
    if (zero == nullptr) {
        zero = initialize<matrix::Dense<ValueType>>({gko::zero<ValueType>()},
                                                    exec);
    }
    if (one == nullptr) {
        one =
            initialize<matrix::Dense<ValueType>>({gko::one<ValueType>()}, exec);
    }
}


}  // namespace


template <typename ValueType>
Combination<ValueType>& Combination<ValueType>::operator=(
    const Combination& other)
{
    if (&other != this) {
        LinOp::operator=(other);
        auto exec = this->get_executor();
        coefficients_ = other.coefficients_;
        operators_ = other.operators_;
        // if the operators are on the wrong executor, copy them over
        if (other.get_executor() != exec) {
            for (auto& coef : coefficients_) {
                coef = gko::clone(exec, coef);
            }
            for (auto& op : operators_) {
                op = gko::clone(exec, op);
            }
        }
    }
    return *this;
}


template <typename ValueType>
Combination<ValueType>& Combination<ValueType>::operator=(Combination&& other)
{
    if (&other != this) {
        LinOp::operator=(std::move(other));
        auto exec = this->get_executor();
        coefficients_ = std::move(other.coefficients_);
        operators_ = std::move(other.operators_);
        // if the operators are on the wrong executor, copy them over
        if (other.get_executor() != exec) {
            for (auto& coef : coefficients_) {
                coef = gko::clone(exec, coef);
            }
            for (auto& op : operators_) {
                op = gko::clone(exec, op);
            }
        }
    }
    return *this;
}


template <typename ValueType>
Combination<ValueType>::Combination(const Combination& other)
    : Combination(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Combination<ValueType>::Combination(Combination&& other)
    : Combination(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
std::unique_ptr<LinOp> Combination<ValueType>::transpose() const
{
    auto transposed = Combination<ValueType>::create(this->get_executor());
    transposed->set_size(gko::transpose(this->get_size()));
    // copy coefficients
    for (auto& coef : get_coefficients()) {
        transposed->coefficients_.push_back(share(coef->clone()));
    }
    // transpose operators
    for (auto& op : get_operators()) {
        transposed->operators_.push_back(
            share(as<Transposable>(op)->transpose()));
    }

    return std::move(transposed);
}


template <typename ValueType>
std::unique_ptr<LinOp> Combination<ValueType>::conj_transpose() const
{
    auto transposed = Combination<ValueType>::create(this->get_executor());
    transposed->set_size(gko::transpose(this->get_size()));
    // conjugate coefficients!
    for (auto& coef : get_coefficients()) {
        transposed->coefficients_.push_back(
            share(as<matrix::Dense<ValueType>>(coef)->conj_transpose()));
    }
    // conjugate-transpose operators
    for (auto& op : get_operators()) {
        transposed->operators_.push_back(
            share(as<Transposable>(op)->conj_transpose()));
    }

    return std::move(transposed);
}


template <typename ValueType>
void Combination<ValueType>::apply_impl(const MultiVector* b,
                                        MultiVector* x) const
{
    initialize_scalars(this->get_executor(), cache_.zero, cache_.one);

    auto converted_b = b->as_precision(this);
    auto converted_x = x->as_precision(this);
    auto dense_b = converted_b.get();
    auto dense_x = converted_x.get();
    operators_[0]->apply(coefficients_[0], dense_b, cache_.zero, dense_x);
    for (size_type i = 1; i < operators_.size(); ++i) {
        operators_[i]->apply(coefficients_[i], dense_b, cache_.one, dense_x);
    }
}


template <typename ValueType>
void Combination<ValueType>::apply_impl(const MultiVector* alpha,
                                        const MultiVector* b,
                                        const MultiVector* beta,
                                        MultiVector* x) const
{
    auto converted_b = b->as_precision(this);
    auto converted_x = x->as_precision(this);
    auto dense_b = converted_b.get();
    auto dense_x = converted_x.get();
    if (cache_.intermediate_x == nullptr ||
        cache_.intermediate_x->get_size() != dense_x->get_size()) {
        cache_.intermediate_x = dense_x->clone();
    }
    this->apply_impl(dense_b, cache_.intermediate_x.get());
    dense_x->scale(beta);
    dense_x->add_scaled(alpha, cache_.intermediate_x);
}


#define GKO_DECLARE_COMBINATION(ValueType) class Combination<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMBINATION);


}  // namespace gko
