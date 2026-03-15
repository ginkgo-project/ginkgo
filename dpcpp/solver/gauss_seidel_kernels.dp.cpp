// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/types.hpp>

#include "core/base/mixed_precision_types.hpp"

namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Gauss Seidel solver namespace.
 *
 * @ingroup gssdl
 */
namespace gssdl {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_ell(std::shared_ptr<const DpcppExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Ell<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void multicolor_fgs_amp(std::shared_ptr<const DpcppExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::AMP<MatrixValueType, IndexType>* const a,
                        const matrix::Dense<InputValueType>* const b,
                        matrix::Dense<OutputValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_BASE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_AMP_KERNEL);


}  // namespace gssdl
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
