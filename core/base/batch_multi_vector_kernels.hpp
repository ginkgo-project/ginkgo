// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_BATCH_MULTI_VECTOR_KERNELS_HPP_
#define GKO_CORE_BASE_BATCH_MULTI_VECTOR_KERNELS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL(_type)  \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const batch::MultiVector<_type>* alpha,      \
               batch::MultiVector<_type>* x)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_ELEMENT_WISE_SCALE_KERNEL(_type)  \
    void element_wise_scale(std::shared_ptr<const DefaultExecutor> exec, \
                            const batch::MultiVector<_type>* alpha,      \
                            batch::MultiVector<_type>* x)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL(_type)  \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const batch::MultiVector<_type>* alpha,      \
                    const batch::MultiVector<_type>* x,          \
                    batch::MultiVector<_type>* y)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL(_type)  \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     const batch::MultiVector<_type>* x,          \
                     const batch::MultiVector<_type>* y,          \
                     batch::MultiVector<_type>* result)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL(_type)  \
    void compute_conj_dot(std::shared_ptr<const DefaultExecutor> exec, \
                          const batch::MultiVector<_type>* x,          \
                          const batch::MultiVector<_type>* y,          \
                          batch::MultiVector<_type>* result)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL(_type)  \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec, \
                       const batch::MultiVector<_type>* x,          \
                       batch::MultiVector<remove_complex<_type>>* result)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL(_type)  \
    void copy(std::shared_ptr<const DefaultExecutor> exec, \
              const batch::MultiVector<_type>* x,          \
              batch::MultiVector<_type>* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL(ValueType);              \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_ELEMENT_WISE_SCALE_KERNEL(ValueType); \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL(ValueType);         \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL(ValueType);        \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_CONJ_DOT_KERNEL(ValueType);   \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL(ValueType);      \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_multi_vector,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_BASE_BATCH_MULTI_VECTOR_KERNELS_HPP_
