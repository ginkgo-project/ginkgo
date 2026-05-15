// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_DIAGONAL_KERNELS_HPP_
#define GKO_CORE_MATRIX_DIAGONAL_KERNELS_HPP_


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL(ValueType)        \
    void apply_to_dense(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Diagonal<ValueType>* a,        \
                        matrix::view::dense<const ValueType> b,      \
                        matrix::view::dense<ValueType> c, bool inverse)

#define GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL(ValueType)        \
    void right_apply_to_dense(std::shared_ptr<const DefaultExecutor> exec, \
                              const matrix::Diagonal<ValueType>* a,        \
                              matrix::view::dense<const ValueType> b,      \
                              matrix::view::dense<ValueType> c)

#define GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL(ValueType, IndexType) \
    void apply_to_csr(std::shared_ptr<const DefaultExecutor> exec,     \
                      const matrix::Diagonal<ValueType>* a,            \
                      const matrix::Csr<ValueType, IndexType>* b,      \
                      matrix::Csr<ValueType, IndexType>* c, bool inverse)

#define GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL(ValueType, IndexType) \
    void right_apply_to_csr(std::shared_ptr<const DefaultExecutor> exec,     \
                            const matrix::Diagonal<ValueType>* a,            \
                            const matrix::Csr<ValueType, IndexType>* b,      \
                            matrix::Csr<ValueType, IndexType>* c)

#define GKO_DECLARE_DIAGONAL_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType) \
    void fill_in_matrix_data(                                                 \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const device_matrix_data<ValueType, IndexType>& data,                 \
        matrix::Diagonal<ValueType>* output)

#define GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL(ValueType, IndexType) \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,     \
                        const matrix::Diagonal<ValueType>* source,       \
                        matrix::view::csr<ValueType, IndexType> result)

#define GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL(ValueType)        \
    void conj_transpose(std::shared_ptr<const DefaultExecutor> exec, \
                        const matrix::Diagonal<ValueType>* orig,     \
                        matrix::Diagonal<ValueType>* trans)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType>                                          \
    GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL(ValueType);                 \
    template <typename ValueType>                                          \
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL(ValueType);           \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL(ValueType, IndexType);  \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIAGONAL_FILL_IN_MATRIX_DATA_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_DIAGONAL_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);      \
    template <typename ValueType>                                          \
    GKO_DECLARE_DIAGONAL_CONJ_TRANSPOSE_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(diagonal, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_DIAGONAL_KERNELS_HPP_
