// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/matrix/identity.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/precision_dispatch.hpp>
#include <ginkgo/core/base/utils.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace matrix {


template <typename ValueType>
void Identity<ValueType>::apply_impl(const LinOp* b, LinOp* x) const
{
    x->copy_from(b);
}


template <typename ValueType>
void Identity<ValueType>::apply_impl(const LinOp* alpha, const LinOp* b,
                                     const LinOp* beta, LinOp* x) const
{
    experimental::precision_dispatch_real_complex_distributed<ValueType>(
        [this](auto dense_alpha, auto dense_b, auto dense_beta, auto dense_x) {
            dense_x->scale(dense_beta);
            dense_x->add_scaled(dense_alpha, dense_b);
        },
        alpha, b, beta, x);
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


#define GKO_DECLARE_IDENTITY_MATRIX(_type) class Identity<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDENTITY_MATRIX);
#define GKO_DECLARE_IDENTITY_FACTORY(_type) class IdentityFactory<_type>
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_IDENTITY_FACTORY);


}  // namespace matrix
}  // namespace gko
