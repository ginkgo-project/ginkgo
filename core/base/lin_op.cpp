// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/multivector.hpp>

namespace gko {


void ScaledIdentityAddable::add_scaled_identity(
    ptr_param<const AbstractMultiVector> a,
    ptr_param<const AbstractMultiVector> b)
{
    GKO_ASSERT_IS_SCALAR(a);
    GKO_ASSERT_IS_SCALAR(b);
    auto ae =
        make_temporary_clone(as<PolymorphicObject>(this)->get_executor(), a);
    auto be =
        make_temporary_clone(as<PolymorphicObject>(this)->get_executor(), b);
    add_scaled_identity_impl(ae.get(), be.get());
}


}  // namespace gko
