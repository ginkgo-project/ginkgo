// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "ginkgo/core/matrix/identity.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/dispatch_helper.hpp"


namespace gko {
namespace matrix {


template <typename ValueType>
void Identity<ValueType>::apply_impl(const AbstractMultiVector* b,
                                     AbstractMultiVector* x) const
{
    as<Cloneable>(x)->copy_from(as<Cloneable>(b));
}


template <typename ValueType>
void Identity<ValueType>::apply_impl(const AbstractMultiVector* alpha,
                                     const AbstractMultiVector* b,
                                     const AbstractMultiVector* beta,
                                     AbstractMultiVector* x) const
{
    auto dense_alpha = as<Dense<ValueType>>(alpha->as_precision(this));
    auto dense_beta = as<Dense<ValueType>>(beta->as_precision(this));
    auto converted_x = x->as_precision(this);

    converted_x->scale(dense_beta.get());
    converted_x->add_scaled(dense_alpha.get(), b->as_precision(this).get());
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
    : LinOp(exec, dim<2>{size}, type_to_precision<ValueType>)
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
