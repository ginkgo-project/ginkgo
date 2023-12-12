// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_ASYNC_JACOBI_KERNELS_HPP_
#define GKO_CORE_SOLVER_ASYNC_JACOBI_KERNELS_HPP_


#include <memory>
#include <string>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/stop/stopping_status.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace async_jacobi {


#define GKO_DECLARE_ASYNC_JACOBI_APPLY_KERNEL(ValueType, IndexType) \
    void apply(std::shared_ptr<const DefaultExecutor> exec,         \
               const std::string& check, int max_iters,             \
               const matrix::Dense<ValueType>* relaxation_factor,   \
               const matrix::Dense<ValueType>* second_factor,       \
               const matrix::Csr<ValueType, IndexType>* a,          \
               const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_ASYNC_JACOBI_APPLY_KERNEL(ValueType, IndexType)


}  // namespace async_jacobi


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(async_jacobi,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_ASYNC_JACOBI_KERNELS_HPP_
