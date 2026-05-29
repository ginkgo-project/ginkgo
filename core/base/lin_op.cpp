// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>

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


void LinOp::set_precision(precision p) noexcept { value_t_ = p; }


}  // namespace gko
