// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

namespace gko {
namespace kernels {
namespace hip {
namespace batch_template {
namespace batch_user {

template <typename ValueType, typename UserOpView>
void apply(std::shared_ptr<const DefaultExecutor> exec, const UserOpView mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x) GKO_NOT_IMPLEMENTED;
}  // namespace batch_user
}  // namespace batch_template
}  // namespace hip
}  // namespace kernels
}  // namespace gko
