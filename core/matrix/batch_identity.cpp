// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_identity.hpp>


#include <algorithm>
#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/batch_dim.hpp>
#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/identity.hpp>


namespace gko {
namespace batch {
namespace matrix {


template <typename ValueType>
Identity<ValueType>::Identity(std::shared_ptr<const Executor> exec,
                              const batch_dim<2>& size)
    : EnableBatchLinOp<Identity<ValueType>>(exec, size)
{
    GKO_ASSERT_BATCH_HAS_SQUARE_DIMENSIONS(this->get_size());
}


template <typename ValueType>
Identity<ValueType>* Identity<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x)
{
    static_cast<const Identity*>(this)->apply(b, x);
    return this;
}


template <typename ValueType>
const Identity<ValueType>* Identity<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->validate_application_parameters(b.get(), x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
Identity<ValueType>* Identity<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x)
{
    static_cast<const Identity*>(this)->apply(alpha, b, beta, x);
    return this;
}


template <typename ValueType>
const Identity<ValueType>* Identity<ValueType>::apply(
    ptr_param<const MultiVector<ValueType>> alpha,
    ptr_param<const MultiVector<ValueType>> b,
    ptr_param<const MultiVector<ValueType>> beta,
    ptr_param<MultiVector<ValueType>> x) const
{
    this->validate_application_parameters(alpha.get(), b.get(), beta.get(),
                                          x.get());
    auto exec = this->get_executor();
    this->apply_impl(make_temporary_clone(exec, alpha).get(),
                     make_temporary_clone(exec, b).get(),
                     make_temporary_clone(exec, beta).get(),
                     make_temporary_clone(exec, x).get());
    return this;
}


template <typename ValueType>
void Identity<ValueType>::apply_impl(const MultiVector<ValueType>* b,
                                     MultiVector<ValueType>* x) const
{
    x->copy_from(b);
}


template <typename ValueType>
void Identity<ValueType>::apply_impl(const MultiVector<ValueType>* alpha,
                                     const MultiVector<ValueType>* b,
                                     const MultiVector<ValueType>* beta,
                                     MultiVector<ValueType>* x) const
{
    x->scale(beta);
    x->add_scaled(alpha, b);
}


#define GKO_DECLARE_BATCH_IDENTITY_MATRIX(ValueType) class Identity<ValueType>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_IDENTITY_MATRIX);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
