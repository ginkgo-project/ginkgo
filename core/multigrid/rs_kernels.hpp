// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MULTIGRID_RS_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_RS_KERNELS_HPP_


#include <memory>

#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace rs {

#define GKO_DECLARE_RS_CHECK_M_MATRIX_KERNEL(ValueType, IndexType)  \
    void check_m_matrix(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                \
        matrix::view::csr<const ValueType, const IndexType> matrix, \
        array<bool>& is_m_matrix_array)

// `off_diag` is the off-diagonal block of a distributed matrix, i.e. the
// couplings of the local rows to rows owned by other ranks. It only widens
// max_offdiag in the strength-of-connection test, so that the threshold - and
// with it the strength mask of the local block - is the same one the
// non-distributed kernel would compute for the full row. A non-distributed
// matrix passes the empty view returned by `rs::no_off_diag_view`.
#define GKO_DECLARE_RS_COMPUTE_SOC_AND_RUN_RS_KERNEL(ValueType, IndexType) \
    void compute_soc_and_run_rs(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                       \
        matrix::view::csr<const ValueType, const IndexType> A,             \
        matrix::view::csr<const ValueType, const IndexType> off_diag,      \
        double theta, array<bool>& is_strong, array<IndexType>& lambda,    \
        array<IndexType>& cf_marker, IndexType& coarse_dim)

// Turns the given rows into C-points and recomputes the number of C-points.
// Used in the distributed case to force every local row another rank couples
// to into the coarse set. Promoting an F-point to a C-point is always a valid
// RS splitting: it only enlarges the coarse set, so every remaining F-point
// keeps its strong C-neighbours.
#define GKO_DECLARE_RS_MARK_FORCED_C_POINTS_KERNEL(IndexType)              \
    void mark_forced_c_points(                                             \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_forced, \
        const IndexType* forced_rows, array<IndexType>& cf_marker,         \
        IndexType& coarse_dim)

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
    template <typename IndexType>                                              \
    GKO_DECLARE_RS_MARK_FORCED_C_POINTS_KERNEL(IndexType);                     \
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
