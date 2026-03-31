// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>
#include <ginkgo/core/base/multivector.hpp>

namespace gko {


LinOp::LinOp(LinOp&& other)
    : PolymorphicObject(std::move(other)),
      size_{std::exchange(other.size_, dim<2>{})},
      value_t_(other.value_t_)
{}


LinOp::LinOp(std::shared_ptr<const Executor> exec, const dim<2>& size,
             precision p)
    : PolymorphicObject(exec), size_{size}, value_t_(p)
{}


precision LinOp::get_precision() const noexcept { return value_t_; }


void ScaledIdentityAddable::add_scaled_identity(ptr_param<const MultiVector> a,
                                                ptr_param<const MultiVector> b)
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
