// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL(ValueType,   \
                                                               IndexType)   \
    void add_diagonal_elements(std::shared_ptr<const DefaultExecutor> exec, \
                               matrix::Csr<ValueType, IndexType>* mtx,      \
                               bool is_sorted)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL(ValueType, \
                                                                 IndexType) \
    void initialize_row_ptrs_l_u(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::Csr<ValueType, IndexType>* system_matrix,             \
        IndexType* l_row_ptrs, IndexType* u_row_ptrs)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL(ValueType, IndexType) \
    void initialize_l_u(                                                      \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const matrix::Csr<ValueType, IndexType>* system_matrix,               \
        matrix::Csr<ValueType, IndexType>* l_factor,                          \
        matrix::Csr<ValueType, IndexType>* u_factor)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL(ValueType, \
                                                               IndexType) \
    void initialize_row_ptrs_l(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const matrix::Csr<ValueType, IndexType>* system_matrix,           \
        IndexType* l_row_ptrs)

#define GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL(ValueType, IndexType)   \
    void initialize_l(std::shared_ptr<const DefaultExecutor> exec,            \
                      const matrix::Csr<ValueType, IndexType>* system_matrix, \
                      matrix::Csr<ValueType, IndexType>* l_factor,            \
                      bool diag_sqrt)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                       \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_ELEMENTS_KERNEL(ValueType,      \
                                                           IndexType);     \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_U_KERNEL(ValueType,    \
                                                             IndexType);   \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_U_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_L_KERNEL(ValueType,      \
                                                           IndexType);     \
    template <typename ValueType, typename IndexType>                      \
    GKO_DECLARE_FACTORIZATION_INITIALIZE_L_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(factorization,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_FACTORIZATION_KERNELS_HPP_
