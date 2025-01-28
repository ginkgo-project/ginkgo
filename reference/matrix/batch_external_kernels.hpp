// SPDX-FileCopyrightText: 2024 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once


#include <algorithm>

#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_external.hpp>

#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "reference/base/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace batch_single_kernels {


template <typename ValueType>
void simple_apply_kernel(
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    a.simple_apply(a.batch_id,
                   {static_cast<size_type>(a.num_rows),
                    static_cast<size_type>(a.num_cols)},
                   b.values, c.values, a.payload);
}


template <typename ValueType>
void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::external::batch_item<const ValueType>& a,
    const gko::batch::multi_vector::batch_item<const ValueType>& b,
    const ValueType beta,
    const gko::batch::multi_vector::batch_item<ValueType>& c)
{
    a.advanced_apply(a.batch_id,
                     {static_cast<size_type>(a.num_rows),
                      static_cast<size_type>(a.num_cols)},
                     &alpha, b.values, &beta, c.values, a.payload);
}


}  // namespace batch_single_kernels
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
