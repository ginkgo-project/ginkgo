// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/base/perturbation.hpp"

#include <ginkgo/core/matrix/dense.hpp>


namespace gko {

template <typename ValueType>
Perturbation<ValueType>& Perturbation<ValueType>::operator=(
    const Perturbation& other)
{
    if (&other != this) {
        LinOp::operator=(other);
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
        LinOp::operator=(std::move(other));
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
Perturbation<ValueType>::Perturbation(std::shared_ptr<const Executor> exec)
    : LinOp(std::move(exec), dim<2>{}, type_to_precision<ValueType>)
{}


template <typename ValueType>
Perturbation<ValueType>::Perturbation(
    std::shared_ptr<const matrix::Dense<ValueType>> scalar,
    std::shared_ptr<const LinOp> basis)
    : Perturbation(std::move(scalar),
                   // basis can not be std::move(basis). Otherwise, Program
                   // deletes basis before applying conjugate transpose
                   basis,
                   std::move((as<gko::Transposable>(basis))->conj_transpose()))
{}


template <typename ValueType>
Perturbation<ValueType>::Perturbation(
    std::shared_ptr<const matrix::Dense<ValueType>> scalar,
    std::shared_ptr<const LinOp> basis, std::shared_ptr<const LinOp> projector)
    : LinOp(basis->get_executor(), gko::dim<2>{basis->get_size()[0]},
            type_to_precision<ValueType>),
      basis_{std::move(basis)},
      projector_{std::move(projector)},
      scalar_{std::move(scalar)}
{
    this->validate_perturbation();
}


template <typename ValueType>
std::unique_ptr<Perturbation<ValueType>> Perturbation<ValueType>::create(
    std::shared_ptr<const Executor> exec)
{
    return std::unique_ptr<Perturbation>{new Perturbation{exec}};
}


template <typename ValueType>
std::unique_ptr<Perturbation<ValueType>> Perturbation<ValueType>::create(
    std::shared_ptr<const matrix::Dense<ValueType>> scalar,
    std::shared_ptr<const LinOp> basis)
{
    return std::unique_ptr<Perturbation>{new Perturbation{scalar, basis}};
}


template <typename ValueType>
std::unique_ptr<Perturbation<ValueType>> Perturbation<ValueType>::create(
    std::shared_ptr<const matrix::Dense<ValueType>> scalar,
    std::shared_ptr<const LinOp> basis, std::shared_ptr<const LinOp> projector)
{
    return std::unique_ptr<Perturbation>{
        new Perturbation{scalar, basis, projector}};
}


template <typename ValueType>
void Perturbation<ValueType>::validate_perturbation()
{
    GKO_ASSERT_CONFORMANT(basis_, projector_);
    GKO_ASSERT_CONFORMANT(projector_, basis_);
    GKO_ASSERT_EQUAL_DIMENSIONS(scalar_, dim<2>(1, 1));
}


template <typename ValueType>
void Perturbation<ValueType>::cache_struct::allocate(
    std::shared_ptr<const Executor> exec, dim<2> size)

{
    using vec = gko::matrix::Dense<ValueType>;
    if (one == nullptr) {
        one = initialize<vec>({gko::one<ValueType>()}, exec);
    }
    if (alpha_scalar == nullptr) {
        alpha_scalar = vec::create(exec, gko::dim<2>(1));
    }
    if (intermediate == nullptr || intermediate->get_size() != size) {
        intermediate = vec::create(exec, size);
    }
}


template <typename ValueType>
void Perturbation<ValueType>::apply_impl(const MultiVector* b,
                                         MultiVector* x) const
{
    // x = (I + scalar * basis * projector) * b
    // temp = projector * b                 : projector->apply(b, temp)
    // x = b                                : x->copy_from(b)
    // x = 1 * x + scalar * basis * temp    : basis->apply(scalar, temp, 1, x)
    auto converted_b = b->as_precision(this);
    auto converted_x = x->as_precision(this);
    auto dense_b = converted_b.get();
    auto dense_x = converted_x.get();
    auto exec = this->get_executor();
    auto intermediate_size =
        gko::dim<2>(projector_->get_size()[0], dense_b->get_size()[1]);
    cache_.allocate(exec, intermediate_size);
    projector_->apply(dense_b, cache_.intermediate);
    dense_x->copy_from(dense_b);
    basis_->apply(scalar_, cache_.intermediate, cache_.one, dense_x);
}


template <typename ValueType>
void Perturbation<ValueType>::apply_impl(const MultiVector* alpha,
                                         const MultiVector* b,
                                         const MultiVector* beta,
                                         MultiVector* x) const
{
    // x = alpha * (I + scalar * basis * projector) b + beta * x
    //   = beta * x + alpha * b + alpha * scalar * basis * projector * b
    // temp = projector * b     : projector->apply(b, temp)
    // x = beta * x + alpha * b : x->scale(beta),
    //                            x->add_scaled(alpha, b)
    // x = x + alpha * scalar * basis * temp
    //                          : basis->apply(alpha * scalar, temp, 1, x)
    auto converted_alpha = alpha->as_precision(this);
    auto converted_b = b->as_precision(this);
    auto converted_x = x->as_precision(this);
    auto dense_alpha = as<matrix::Dense<ValueType>>(converted_alpha.get());
    auto dense_b = converted_b.get();
    auto dense_x = converted_x.get();

    auto exec = this->get_executor();
    auto intermediate_size =
        gko::dim<2>(projector_->get_size()[0], dense_b->get_size()[1]);
    cache_.allocate(exec, intermediate_size);
    projector_->apply(dense_b, cache_.intermediate);
    dense_x->scale(beta);
    dense_x->add_scaled(dense_alpha, dense_b);
    cache_.alpha_scalar->copy_from(dense_alpha);
    cache_.alpha_scalar->scale(scalar_);
    basis_->apply(cache_.alpha_scalar, cache_.intermediate, cache_.one,
                  dense_x);
}


#define GKO_DECLARE_PERTURBATION(ValueType) class Perturbation<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_PERTURBATION);


}  // namespace gko
