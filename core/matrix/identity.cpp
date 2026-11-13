// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/identity.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
void Identity<ValueType>::apply_impl(const AbstractMultiVector* b,
                                     AbstractMultiVector* x) const
{
    x->copy_from(b);
}


template <typename ValueType>
void Identity<ValueType>::apply_impl(const AbstractMultiVector* alpha,
                                     const AbstractMultiVector* b,
                                     const AbstractMultiVector* beta,
                                     AbstractMultiVector* x) const
{
    precision_dispatch<ValueType>(
        [alpha, beta](auto b_, auto x_) {
            x_->scale(beta);
            x_->add_scaled(alpha, b_);
        },
        b, x);
}


template <typename ValueType>
std::unique_ptr<LinOp> IdentityFactory<ValueType>::generate_impl(
    std::shared_ptr<const LinOp> base) const
{
    GKO_ASSERT_EQUAL_DIMENSIONS(base, transpose(base->get_size()));
    return Identity<ValueType>::create(this->get_executor(),
                                       base->get_size()[0]);
}


template <typename ValueType>
std::unique_ptr<LinOp> Identity<ValueType>::transpose() const
{
    return this->clone();
}


template <typename ValueType>
std::unique_ptr<LinOp> Identity<ValueType>::conj_transpose() const
{
    return this->clone();
}


template <typename ValueType>
Identity<ValueType>::Identity(std::shared_ptr<const Executor> exec,
                              size_type size)
    : LinOp(exec, dim<2>{size})
{}


template <typename ValueType>
std::unique_ptr<Identity<ValueType>> Identity<ValueType>::create(
    std::shared_ptr<const Executor> exec, dim<2> size)
{
    GKO_ASSERT_IS_SQUARE_MATRIX(size);
    return std::unique_ptr<Identity>{new Identity{exec, size[0]}};
}


template <typename ValueType>
std::unique_ptr<Identity<ValueType>> Identity<ValueType>::create(
    std::shared_ptr<const Executor> exec, size_type size)
{
    return std::unique_ptr<Identity>{new Identity{exec, size}};
}


#define GKO_DECLARE_IDENTITY_MATRIX(ValueType) class Identity<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDENTITY_MATRIX);
#define GKO_DECLARE_IDENTITY_FACTORY(ValueType) class IdentityFactory<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDENTITY_FACTORY);


}  // namespace matrix
}  // namespace gko
