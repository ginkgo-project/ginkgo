// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_ILUT_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_ILUT_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/ilut.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {

#define GKO_DECLARE_ILUT_INITIALIZE_KERNEL(ValueType, IndexType)        \
    void initialize(                                                    \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* mtx,                   \
        matrix::Csr<ValueType, IndexType>* l_factor,                    \
        const IndexType* l_lookup_offsets, const int64* l_lookup_descs, \
        const int32* l_lookup_storage,                                  \
        matrix::Csr<ValueType, IndexType>* u_factor,                    \
        const IndexType* u_lookup_offsets, const int64* u_lookup_descs, \
        const int32* u_lookup_storage)

#define GKO_DECLARE_ILUT_COMPUTE_LU_FACTORS_KERNEL(ValueType, IndexType)     \
    void compute_l_u_factors(                                                \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        matrix::Csr<ValueType, IndexType>* l,                                \
        const IndexType* l_lookup_offsets, const int64* l_lookup_descs,      \
        const int32* l_lookup_storage, matrix::Csr<ValueType, IndexType>* u, \
        const IndexType* u_lookup_offsets, const int64* u_lookup_descs,      \
        const int32* u_lookup_storage, array<int>& tmp_storage)

#define GKO_DECLARE_ALL_AS_TEMPLATES                          \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_ILUT_INITIALIZE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>         \
    GKO_DECLARE_ILUT_COMPUTE_LU_FACTORS_KERNEL(ValueType, IndexType);


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(ilut_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_
