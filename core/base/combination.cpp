// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/combination.hpp"

#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "dispatch_helper.hpp"


namespace gko {
namespace {


template <typename ValueType>
inline void initialize_scalars(
    std::shared_ptr<const Executor> exec,
    std::unique_ptr<matrix::MultiVector<ValueType>>& zero,
    std::unique_ptr<matrix::MultiVector<ValueType>>& one)
{
    if (zero == nullptr) {
        zero = initialize<matrix::MultiVector<ValueType>>(
            {gko::zero<ValueType>()}, exec);
    }
    if (one == nullptr) {
        one = initialize<matrix::MultiVector<ValueType>>(
            {gko::one<ValueType>()}, exec);
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
            share(as<matrix::MultiVector<ValueType>>(coef)->conj_transpose()));
    }
    // conjugate-transpose operators
    for (auto& op : get_operators()) {
        transposed->operators_.push_back(
            share(as<Transposable>(op)->conj_transpose()));
    }

    return std::move(transposed);
}


template <typename ValueType>
void Combination<ValueType>::apply_impl(const AbstractMultiVector* b,
                                        AbstractMultiVector* x) const
{
    initialize_scalars(this->get_executor(), cache_.zero, cache_.one);

    precision_dispatch<ValueType>(
        [this](auto b_, auto x_) {
            operators_[0]->apply(coefficients_[0], b_, cache_.zero, x_);
            for (size_type i = 1; i < operators_.size(); ++i) {
                operators_[i]->apply(coefficients_[i], b_, cache_.one, x_);
            }
        },
        b, x);
}


template <typename ValueType>
void Combination<ValueType>::apply_impl(const AbstractMultiVector* alpha,
                                        const AbstractMultiVector* b,
                                        const AbstractMultiVector* beta,
                                        AbstractMultiVector* x) const
{
    precision_dispatch<ValueType>(
        [this, alpha, beta](auto b_, auto x_) {
            if (cache_.intermediate_x == nullptr ||
                cache_.intermediate_x->get_size() != x_->get_size()) {
                cache_.intermediate_x = x_->clone();
            }
            this->apply_impl(b_, cache_.intermediate_x.get());
            x_->scale(beta);
            x_->add_scaled(alpha, cache_.intermediate_x);
        },
        b, x);
}


#define GKO_DECLARE_COMBINATION(ValueType) class Combination<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_COMBINATION);


}  // namespace gko
