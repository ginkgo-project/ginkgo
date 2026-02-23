// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/lin_op.hpp>

namespace gko {


bool LinOp::supports_mixed_precision() const noexcept { return false; }


precision LinOp::get_precision() const noexcept { return value_t_; }


}  // namespace gko
