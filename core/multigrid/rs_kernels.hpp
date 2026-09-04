// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_RS_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_RS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/multivector.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace rs {

#define GKO_DECLARE_RS_CHECK_M_MATRIX_KERNEL(ValueType, IndexType)  \
    void check_m_matrix(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                \
        matrix::view::csr<const ValueType, const IndexType> matrix, \
        array<bool>& is_m_matrix_array)

#define GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL(ValueType, IndexType)   \
    void compute_soc_and_run_rs(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        matrix::view::csr<const ValueType, const IndexType> A, double theta, \
        array<bool>& is_strong, array<IndexType>& lambda,                    \
        array<IndexType>& cf_marker, IndexType& coarse_dim)

#define GKO_DECLARE_RS_FILL_COARSE_AND_COMPUTE_PROLONG_ROW_PTRS_KERNEL(   \
    ValueType, IndexType)                                                 \
    void fill_coarse_and_compute_prolong_row_ptrs(                        \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const array<IndexType>& cf_marker, array<IndexType>& coarse_rows, \
        array<IndexType>& fine_to_coarse,                                 \
        matrix::view::csr<const ValueType, const IndexType> A,            \
        const array<bool>& is_strong, array<IndexType>& row_ptrs)

#define GKO_DECLARE_RS_COMPUTE_INTERPOLATION_KERNEL(ValueType, IndexType) \
    void compute_interpolation(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        matrix::view::csr<const ValueType, const IndexType> A,            \
        const bool* is_strong, const array<IndexType>& cf_marker,         \
        const IndexType* fine_to_coarse,                                  \
        matrix::view::csr<ValueType, IndexType> P)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                           \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_RS_CHECK_M_MATRIX_KERNEL(ValueType, IndexType);                \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL(ValueType, IndexType);        \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_RS_FILL_COARSE_AND_COMPUTE_PROLONG_ROW_PTRS_KERNEL(ValueType,  \
                                                                   IndexType); \
    template <typename ValueType, typename IndexType>                          \
    GKO_DECLARE_RS_COMPUTE_INTERPOLATION_KERNEL(ValueType, IndexType)


}  // namespace rs


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(rs, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_RS_KERNELS_HPP_
