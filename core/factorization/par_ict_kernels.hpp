// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_PAR_ICT_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ICT_KERNELS_HPP_


#include <ginkgo/core/factorization/par_ict.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL(ValueType, IndexType) \
    void add_candidates(std::shared_ptr<const DefaultExecutor> exec,    \
                        const matrix::Csr<ValueType, IndexType>* llh,   \
                        const matrix::Csr<ValueType, IndexType>* a,     \
                        const matrix::Csr<ValueType, IndexType>* l,     \
                        matrix::Csr<ValueType, IndexType>* l_new)

#define GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL(ValueType, IndexType) \
    void compute_factor(std::shared_ptr<const DefaultExecutor> exec,    \
                        const matrix::Csr<ValueType, IndexType>* a,     \
                        matrix::Csr<ValueType, IndexType>* l,           \
                        const matrix::Coo<ValueType, IndexType>* l_coo)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                 \
    template <typename ValueType, typename IndexType>                \
    GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                \
    GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(par_ict_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ICT_KERNELS_HPP_
