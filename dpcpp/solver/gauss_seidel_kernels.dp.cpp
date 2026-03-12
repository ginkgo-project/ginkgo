// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/gauss_seidel_kernels.hpp"

#include "ginkgo/core/base/exception_helpers.hpp"
#include "ginkgo/core/base/types.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Gauss Seidel solver namespace.
 *
 * @ingroup gssdl
 */
namespace gssdl {


template <typename ValueType, typename IndexType>
void multicolor_fgs_ell(std::shared_ptr<const DpcppExecutor> exec,
                        const std::vector<IndexType>& color_ptrs,
                        const matrix::Ell<ValueType, IndexType>* const a,
                        const matrix::Dense<ValueType>* const b,
                        matrix::Dense<ValueType>* const x,
                        const bool first_iter,
                        array<stopping_status>* const stop_status)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL);


}  // namespace gssdl
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
