// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_PAR_ILU_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILU_KERNELS_HPP_


#include <ginkgo/core/factorization/par_ilu.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {

#define GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL(ValueType, IndexType) \
    void compute_l_u_factors(                                                \
        std::shared_ptr<const DefaultExecutor> exec, size_type iterations,   \
        const matrix::Coo<ValueType, IndexType>* system_matrix,              \
        matrix::Csr<ValueType, IndexType>* l_factor,                         \
        matrix::Csr<ValueType, IndexType>* u_factor)


#define GKO_DECLARE_ALL_AS_TEMPLATES                  \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(par_ilu_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILU_KERNELS_HPP_
