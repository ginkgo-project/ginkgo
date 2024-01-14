// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_BATCH_ELL_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_ELL_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_ell.hpp>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_ELL_SIMPLE_APPLY_KERNEL(_vtype, _itype)  \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const batch::matrix::Ell<_vtype, _itype>* a, \
                      const batch::MultiVector<_vtype>* b,         \
                      batch::MultiVector<_vtype>* c)

#define GKO_DECLARE_BATCH_ELL_ADVANCED_APPLY_KERNEL(_vtype, _itype)  \
    void advanced_apply(std::shared_ptr<const DefaultExecutor> exec, \
                        const batch::MultiVector<_vtype>* alpha,     \
                        const batch::matrix::Ell<_vtype, _itype>* a, \
                        const batch::MultiVector<_vtype>* b,         \
                        const batch::MultiVector<_vtype>* beta,      \
                        batch::MultiVector<_vtype>* c)

#define GKO_DECLARE_BATCH_ELL_SCALE_KERNEL(_vtype, _itype)  \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const array<_vtype>* left_scale,             \
               const array<_vtype>* right_scale,            \
               batch::matrix::Ell<_vtype, _itype>* input)

#define GKO_DECLARE_BATCH_ELL_ADD_SCALED_IDENTITY_KERNEL(_vtype, _itype)  \
    void add_scaled_identity(std::shared_ptr<const DefaultExecutor> exec, \
                             const batch::MultiVector<_vtype>* alpha,     \
                             const batch::MultiVector<_vtype>* beta,      \
                             batch::matrix::Ell<_vtype, _itype>* mat)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                   \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_BATCH_ELL_SIMPLE_APPLY_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_BATCH_ELL_ADVANCED_APPLY_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_BATCH_ELL_SCALE_KERNEL(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>                  \
    GKO_DECLARE_BATCH_ELL_ADD_SCALED_IDENTITY_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_ell,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_ELL_KERNELS_HPP_
