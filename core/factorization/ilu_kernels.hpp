// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_ILU_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_ILU_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/factorization/ilu.hpp>
#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_ILU_SPARSELIB_ILU_KERNEL(ValueType, IndexType)  \
    void sparselib_ilu(std::shared_ptr<const DefaultExecutor> exec, \
                       matrix::Csr<ValueType, IndexType>* system_matrix)
#define GKO_DECLARE_ILU_FACTORIZE_ON_BOTH_KERNEL(ValueType, IndexType)        \
    void factorize_on_both(                                                   \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const IndexType* lookup_offsets, const int64* lookup_descs,           \
        const int32* lookup_storage, const IndexType* diag_idxs,              \
        matrix::Csr<ValueType, IndexType>* factors,                           \
        const IndexType* matrix_lookup_offsets,                               \
        const int64* matrix_lookup_descs, const int32* matrix_lookup_storage, \
        matrix::Csr<ValueType, IndexType>* matrix, array<int>& tmp_storage)


#define GKO_DECLARE_ALL_AS_TEMPLATES                            \
    template <typename ValueType, typename IndexType>           \
    GKO_DECLARE_ILU_SPARSELIB_ILU_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>           \
    GKO_DECLARE_ILU_FACTORIZE_ON_BOTH_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(ilu_factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_ILU_KERNELS_HPP_
