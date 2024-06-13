// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_PRECONDITIONER_ISAI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_ISAI_KERNELS_HPP_


#include <ginkgo/core/preconditioner/isai.hpp>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL(ValueType, IndexType)    \
    void generate_tri_inverse(std::shared_ptr<const DefaultExecutor> exec,    \
                              const matrix::Csr<ValueType, IndexType>* input, \
                              matrix::Csr<ValueType, IndexType>* inverse,     \
                              IndexType* excess_rhs_ptrs,                     \
                              IndexType* excess_nz_ptrs, bool lower)

#define GKO_DECLARE_ISAI_GENERATE_GENERAL_INVERSE_KERNEL(ValueType, IndexType) \
    void generate_general_inverse(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                           \
        const matrix::Csr<ValueType, IndexType>* input,                        \
        matrix::Csr<ValueType, IndexType>* inverse,                            \
        IndexType* excess_rhs_ptrs, IndexType* excess_nz_ptrs, bool spd)

#define GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL(ValueType, IndexType) \
    void generate_excess_system(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Csr<ValueType, IndexType>* input,                      \
        const matrix::Csr<ValueType, IndexType>* inverse,                    \
        const IndexType* excess_rhs_ptrs, const IndexType* excess_nz_ptrs,   \
        matrix::Csr<ValueType, IndexType>* excess_system,                    \
        matrix::Dense<ValueType>* excess_rhs, size_type e_start,             \
        size_type e_end)

#define GKO_DECLARE_ISAI_SCALE_EXCESS_SOLUTION_KERNEL(ValueType, IndexType) \
    void scale_excess_solution(std::shared_ptr<const DefaultExecutor> exec, \
                               const IndexType* excess_block_ptrs,          \
                               matrix::Dense<ValueType>* excess_solution,   \
                               size_type e_start, size_type e_end)

#define GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL(ValueType, IndexType) \
    void scatter_excess_solution(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                          \
        const IndexType* excess_rhs_ptrs,                                     \
        const matrix::Dense<ValueType>* excess_solution,                      \
        matrix::Csr<ValueType, IndexType>* inverse, size_type e_start,        \
        size_type e_end)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    constexpr int row_size_limit = 32;                                      \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_ISAI_GENERATE_GENERAL_INVERSE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_ISAI_SCALE_EXCESS_SOLUTION_KERNEL(ValueType, IndexType);    \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(isai, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_ISAI_KERNELS_HPP_
