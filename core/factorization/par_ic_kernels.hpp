// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_PAR_IC_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_PAR_IC_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/par_ic.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/device_views.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL(ValueType, IndexType) \
    void init_factor(std::shared_ptr<const DefaultExecutor> exec,   \
                     matrix::view::csr<ValueType, IndexType> l_factor)

#define GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL(ValueType, IndexType)      \
    void compute_factor(std::shared_ptr<const DefaultExecutor> exec,        \
                        size_type iterations,                               \
                        matrix::view::coo<const ValueType, const IndexType> \
                            lower_system_matrix,                            \
                        matrix::view::csr<ValueType, IndexType> l_factor)

#define GKO_DECLARE_ALL_AS_TEMPLATES                             \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>            \
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(par_ic_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_IC_KERNELS_HPP_
