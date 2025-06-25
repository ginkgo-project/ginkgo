// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SOLVER_CHEBYSHEV_KERNELS_HPP_
#define GKO_CORE_SOLVER_CHEBYSHEV_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/solver/chebyshev.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace chebyshev {


#define GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL(ValueType)             \
    void init_update(std::shared_ptr<const DefaultExecutor> exec,       \
                     const solver::detail::coeff_type<ValueType> alpha, \
                     const matrix::Dense<ValueType>* inner_sol,         \
                     matrix::Dense<ValueType>* update_sol,              \
                     matrix::Dense<ValueType>* output)

#define GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL(ValueType)             \
    void update(std::shared_ptr<const DefaultExecutor> exec,       \
                const solver::detail::coeff_type<ValueType> alpha, \
                const solver::detail::coeff_type<ValueType> beta,  \
                matrix::Dense<ValueType>* inner_sol,               \
                matrix::Dense<ValueType>* update_sol,              \
                matrix::Dense<ValueType>* output)

#define GKO_DECLARE_ALL_AS_TEMPLATES                     \
    template <typename ValueType>                        \
    GKO_DECLARE_CHEBYSHEV_INIT_UPDATE_KERNEL(ValueType); \
    template <typename ValueType>                        \
    GKO_DECLARE_CHEBYSHEV_UPDATE_KERNEL(ValueType)


}  // namespace chebyshev


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(chebyshev,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SOLVER_CHEBYSHEV_KERNELS_HPP_
