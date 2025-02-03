// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <ginkgo/config.hpp>

#if GINKGO_BUILD_OMP

#include "batch_csr_kernels.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace batch_template {
namespace batch_user {

template <typename ValueType, typename UserOpView>
void apply(std::shared_ptr<const DefaultExecutor> exec, const UserOpView mat,
           batch::multi_vector::uniform_batch<const ValueType> b,
           batch::multi_vector::uniform_batch<ValueType> x)
{
    const size_type num_batch_items = mat.num_batch_items;
    const auto num_rhs = b.num_rhs;
    if (num_rhs > 1) {
        GKO_NOT_IMPLEMENTED;
    }

#pragma omp parallel for
    for (size_type batch_id = 0; batch_id < num_batch_items; batch_id++) {
        batch_single_kernels::simple_apply(
            batch::extract_batch_item(mat, batch_id),
            batch::extract_batch_item(b, batch_id),
            batch::extract_batch_item(x, batch_id));
    }
}
}  // namespace batch_user
}  // namespace batch_template
}  // namespace omp
}  // namespace kernels
}  // namespace gko

#endif
