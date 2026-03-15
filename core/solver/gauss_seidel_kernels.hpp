// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_GAUSS_SEIDEL_KERNELS_HPP_
#define GKO_CORE_SOLVER_GAUSS_SEIDEL_KERNELS_HPP_


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/amp.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace gssdl {


#define GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL(                             \
    InputValueType, MatrixValueType, OutputValueType, IndexType)              \
    void multicolor_fgs_ell(std::shared_ptr<const DefaultExecutor> exec,      \
                            const std::vector<IndexType>& color_ptrs,         \
                            const matrix::Ell<MatrixValueType, IndexType>* a, \
                            const matrix::Dense<InputValueType>* b,           \
                            matrix::Dense<OutputValueType>* x,                \
                            bool first_iter,                                  \
                            array<stopping_status>* stop_status)


#define GKO_DECLARE_MULTICOLOR_FWD_GS_AMP_KERNEL(                             \
    InputValueType, MatrixValueType, OutputValueType, IndexType)              \
    void multicolor_fgs_amp(std::shared_ptr<const DefaultExecutor> exec,      \
                            const std::vector<IndexType>& color_ptrs,         \
                            const matrix::AMP<MatrixValueType, IndexType>* a, \
                            const matrix::Dense<InputValueType>* b,           \
                            matrix::Dense<OutputValueType>* x,                \
                            bool first_iter,                                  \
                            array<stopping_status>* stop_status)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                          \
    template <typename InputValueType, typename MatrixValueType,              \
              typename OutputValueType, typename IndexType>                   \
    GKO_DECLARE_MULTICOLOR_FWD_GS_ELL_KERNEL(InputValueType, MatrixValueType, \
                                             OutputValueType, IndexType);     \
    template <typename InputValueType, typename MatrixValueType,              \
              typename OutputValueType, typename IndexType>                   \
    GKO_DECLARE_MULTICOLOR_FWD_GS_AMP_KERNEL(InputValueType, MatrixValueType, \
                                             OutputValueType, IndexType)


}  // namespace gssdl


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gssdl, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_GAUSS_SEIDEL_KERNELS_HPP_
