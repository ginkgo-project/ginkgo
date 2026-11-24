// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/isai_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/unified/base/kernel_launch.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace isai {


template <typename ValueType, typename IndexType>
void generate_spd_inverse_rowwise(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::csr<const ValueType, const IndexType> input,
    matrix::view::csr<ValueType, IndexType> inverse,
    size_type workspace_bytes) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_SPD_INVERSE_ROWWISE_KERNEL);


}  // namespace isai
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
