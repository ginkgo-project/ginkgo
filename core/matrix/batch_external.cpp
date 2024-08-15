// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/batch_external.hpp>


namespace gko {
namespace batch {
namespace matrix {


template <typename ValueType>
std::unique_ptr<External<ValueType>> External<ValueType>::create(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
    External::functor_operation<external_apply::simple_type> simple_apply,
    External::functor_operation<external_apply::advanced_type> advanced_apply,
    void* payload)
{
    return std::unique_ptr<External>(new External(
        std::move(exec), size, simple_apply, advanced_apply, payload));
}


template <typename ValueType>
External<ValueType>* External<ValueType>::apply(
    ptr_param<const MultiVector<value_type>> b,
    ptr_param<MultiVector<value_type>> x)
{
    const_cast<const External*>(this)->apply(b, x);
    return this;
}


template <typename ValueType>
External<ValueType>* External<ValueType>::apply(
    ptr_param<const MultiVector<value_type>> alpha,
    ptr_param<const MultiVector<value_type>> b,
    ptr_param<const MultiVector<value_type>> beta,
    ptr_param<MultiVector<value_type>> x)
{
    const_cast<const External*>(this)->apply(alpha, b, beta, x);
    return this;
}


template <typename ValueType>
const External<ValueType>* External<ValueType>::apply(
    ptr_param<const MultiVector<value_type>> b,
    ptr_param<MultiVector<value_type>> x) const
{
    apply_impl(b.get(), x.get());
    return this;
}


template <typename ValueType>
const External<ValueType>* External<ValueType>::apply(
    ptr_param<const MultiVector<value_type>> alpha,
    ptr_param<const MultiVector<value_type>> b,
    ptr_param<const MultiVector<value_type>> beta,
    ptr_param<MultiVector<value_type>> x) const
{
    apply_impl(alpha.get(), b.get(), beta.get(), x.get());
    return this;
}


template <typename ValueType>
External<ValueType>::External(std::shared_ptr<const Executor> exec)
    : EnableBatchLinOp<External<ValueType>>(std::move(exec))
{}


template <typename ValueType>
External<ValueType>::External(
    std::shared_ptr<const Executor> exec, const batch_dim<2>& size,
    External::functor_operation<external_apply::simple_type> simple_apply,
    External::functor_operation<external_apply::advanced_type> advanced_apply,
    void* payload)
    : EnableBatchLinOp<External<ValueType>>(std::move(exec), size),
      simple_apply_(simple_apply),
      advanced_apply_(advanced_apply),
      payload_(payload)
{}


template <typename ValueType>
void External<ValueType>::apply_impl(const MultiVector<ValueType>* b,
                                     MultiVector<ValueType>* x) const
{}


template <typename ValueType>
void External<ValueType>::apply_impl(const MultiVector<ValueType>* alpha,
                                     const MultiVector<ValueType>* b,
                                     const MultiVector<ValueType>* beta,
                                     MultiVector<ValueType>* x) const
{}


#define GKO_DECLARE_BATCH_MATRIX_EXTERNAL(_vtype) class External<_vtype>

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_BATCH_MATRIX_EXTERNAL);


}  // namespace matrix
}  // namespace batch
}  // namespace gko
