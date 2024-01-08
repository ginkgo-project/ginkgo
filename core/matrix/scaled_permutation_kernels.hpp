// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_SCALED_PERMUTATION_KERNELS_HPP_
#define GKO_CORE_MATRIX_SCALED_PERMUTATION_KERNELS_HPP_


#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_SCALED_PERMUTATION_INVERT_KERNEL(ValueType, IndexType) \
    void invert(std::shared_ptr<const DefaultExecutor> exec,               \
                const ValueType* input_scale,                              \
                const IndexType* input_permutation, size_type size,        \
                ValueType* output_scale, IndexType* output_permutation)

#define GKO_DECLARE_SCALED_PERMUTATION_COMPOSE_KERNEL(ValueType, IndexType) \
    void compose(std::shared_ptr<const DefaultExecutor> exec,               \
                 const ValueType* first_scale,                              \
                 const IndexType* first_permutation,                        \
                 const ValueType* second_scale,                             \
                 const IndexType* second_permutation, size_type size,       \
                 ValueType* output_scale, IndexType* output_permutation)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                    \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_SCALED_PERMUTATION_INVERT_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                   \
    GKO_DECLARE_SCALED_PERMUTATION_COMPOSE_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(scaled_permutation,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_SCALED_PERMUTATION_KERNELS_HPP_
