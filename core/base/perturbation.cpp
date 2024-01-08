// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/perturbation.hpp>


#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {

template <typename ValueType>
Perturbation<ValueType>& Perturbation<ValueType>::operator=(
    const Perturbation& other)
{
    if (&other != this) {
        EnableLinOp<Perturbation>::operator=(other);
        auto exec = this->get_executor();
        scalar_ = other.scalar_;
        basis_ = other.basis_;
        projector_ = other.projector_;
        if (other.get_executor() != exec) {
            scalar_ = gko::clone(exec, scalar_);
            basis_ = gko::clone(exec, basis_);
            projector_ = gko::clone(exec, projector_);
        }
    }
    return *this;
}


template <typename ValueType>
Perturbation<ValueType>& Perturbation<ValueType>::operator=(
    Perturbation&& other)
{
    if (&other != this) {
        EnableLinOp<Perturbation>::operator=(std::move(other));
        auto exec = this->get_executor();
        scalar_ = std::move(other.scalar_);
        basis_ = std::move(other.basis_);
        projector_ = std::move(other.projector_);
        if (other.get_executor() != exec) {
            scalar_ = gko::clone(exec, scalar_);
            basis_ = gko::clone(exec, basis_);
            projector_ = gko::clone(exec, projector_);
        }
    }
    return *this;
}


template <typename ValueType>
Perturbation<ValueType>::Perturbation(const Perturbation& other)
    : Perturbation(other.get_executor())
{
    *this = other;
}


template <typename ValueType>
Perturbation<ValueType>::Perturbation(Perturbation&& other)
    : Perturbation(other.get_executor())
{
    *this = std::move(other);
}


template <typename ValueType>
void Perturbation<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    // x = (I + scalar * basis * projector) * b
    // temp = projector * b                 : projector->apply(b, temp)
    // x = b                                : x->copy_from(b)
    // x = 1 * x + scalar * basis * temp    : basis->apply(scalar, temp, 1, x)
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_b, auto dense_x) {
            auto exec = this->get_executor();
            auto intermediate_size =
                gko::dim<2>(projector_->get_size()[0], dense_b->get_size()[1]);
            cache_.allocate(exec, intermediate_size);
            projector_->apply(dense_b, cache_.intermediate);
            dense_x->copy_from(dense_b);
            basis_->apply(scalar_, cache_.intermediate, cache_.one, dense_x);
        },
        b, x);
}


template <typename ValueType>
void Perturbation<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                         const LinOp* beta, LinOp* x) const
{
    // x = alpha * (I + scalar * basis * projector) b + beta * x
    //   = beta * x + alpha * b + alpha * scalar * basis * projector * b
    // temp = projector * b     : projector->apply(b, temp)
    // x = beta * x + alpha * b : x->scale(beta),
    //                            x->add_scaled(alpha, b)
    // x = x + alpha * scalar * basis * temp
    //                          : basis->apply(alpha * scalar, temp, 1, x)
    precision_dispatch_real_complex<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            auto exec = this->get_executor();
            auto intermediate_size =
                gko::dim<2>(projector_->get_size()[0], dense_b->get_size()[1]);
            cache_.allocate(exec, intermediate_size);
            projector_->apply(dense_b, cache_.intermediate);
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, dense_b);
            dense_alpha->apply(scalar_, cache_.alpha_scalar);
            basis_->apply(cache_.alpha_scalar, cache_.intermediate, cache_.one,
                          dense_x);
        },
        alpha, b, beta, x);
}


#define GKO_DECLARE_PERTURBATION(_type) class Perturbation<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PERTURBATION);


}  // namespace gko
