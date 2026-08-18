// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/core/matrix/multivector.hpp>

namespace gko {
namespace matrix {


template <typename ValueType = default_precision>
using Dense = MultiVector<ValueType>;


}
}  // namespace gko
