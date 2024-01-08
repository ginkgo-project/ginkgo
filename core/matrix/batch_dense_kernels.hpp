// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_


#include <ginkgo/core/matrix/batch_dense.hpp>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL(_type)         \
    void simple_apply(std::shared_ptr<const DefaultExecutor> exec, \
                      const batch::matrix::Dense<_type>* a,        \
                      const batch::MultiVector<_type>* b,          \
                      batch::MultiVector<_type>* c)

#define GKO_DECLARE_BATCH_DENSE_ADVANCED_APPLY_KERNEL(_type)         \
    void advanced_apply(std::shared_ptr<const DefaultExecutor> exec, \
                        const batch::MultiVector<_type>* alpha,      \
                        const batch::matrix::Dense<_type>* a,        \
                        const batch::MultiVector<_type>* b,          \
                        const batch::MultiVector<_type>* beta,       \
                        batch::MultiVector<_type>* c)

#define GKO_DECLARE_ALL_AS_TEMPLATES                        \
    template <typename ValueType>                           \
    GKO_DECLARE_BATCH_DENSE_SIMPLE_APPLY_KERNEL(ValueType); \
    template <typename ValueType>                           \
    GKO_DECLARE_BATCH_DENSE_ADVANCED_APPLY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_dense,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_DENSE_KERNELS_HPP_
