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

#define GKO_DECLARE_RS_COMPUTE_SOC_ROW_PTRS_KERNEL(ValueType, IndexType)    \
    void compute_soc_row_ptrs(std::shared_ptr<const DefaultExecutor> exec,  \
                              const matrix::Csr<ValueType, IndexType>* A,   \
                              remove_complex<ValueType> strength_threshold, \
                              IndexType* row_ptrs)

#define GKO_DECLARE_RS_FILL_SOC_KERNEL(ValueType, IndexType) \
    void fill_soc(                                           \
        std::shared_ptr<const DefaultExecutor> exec,         \
        const matrix::Csr<ValueType, IndexType>* A,          \
        remove_complex<ValueType> strength_threshold,        \
        matrix::Csr<ValueType, IndexType>*                   \
            soc)  // I tried using SparsityCsr here but it doesn't support all
                  // the types. Not sure what'd be the difference either way.

#define GKO_DECLARE_RS_COMPUTE_LAMBDA_KERNEL(ValueType, IndexType)    \
    void compute_lambda(std::shared_ptr<const DefaultExecutor> exec,  \
                        const matrix::Csr<ValueType, IndexType>* soc, \
                        IndexType* lambda)

#define GKO_DECLARE_RS_INIT_CF_KERNEL(IndexType)              \
    void init_cf(std::shared_ptr<const DefaultExecutor> exec, \
                 array<IndexType>& cf_marker)

#define GKO_DECLARE_RS_COARSENING_KERNEL(ValueType, IndexType)       \
    void rs_coarsening(std::shared_ptr<const DefaultExecutor> exec,  \
                       const matrix::Csr<ValueType, IndexType>* soc, \
                       IndexType* lambda, array<IndexType>& cf_marker)

#define GKO_DECLARE_RS_CLEANUP_KERNEL(IndexType)                 \
    void rs_cleanup(std::shared_ptr<const DefaultExecutor> exec, \
                    array<IndexType>& cf_marker)

#define GKO_DECLARE_RS_COUNT_COARSE_KERNEL(IndexType)              \
    void count_coarse(std::shared_ptr<const DefaultExecutor> exec, \
                      const array<IndexType>& cf_marker,           \
                      IndexType* coarse_dim)

#define GKO_DECLARE_RS_FILL_COARSE_ROWS_KERNEL(IndexType)              \
    void fill_coarse_rows(std::shared_ptr<const DefaultExecutor> exec, \
                          const array<IndexType>& cf_marker,           \
                          IndexType* coarse_rows)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_RS_COMPUTE_SOC_ROW_PTRS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_RS_FILL_SOC_KERNEL(ValueType, IndexType);             \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_RS_COMPUTE_LAMBDA_KERNEL(ValueType, IndexType);       \
    template <typename IndexType>                                     \
    GKO_DECLARE_RS_INIT_CF_KERNEL(IndexType);                         \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_RS_COARSENING_KERNEL(ValueType, IndexType);           \
    template <typename IndexType>                                     \
    GKO_DECLARE_RS_CLEANUP_KERNEL(IndexType);                         \
    template <typename IndexType>                                     \
    GKO_DECLARE_RS_COUNT_COARSE_KERNEL(IndexType);                    \
    template <typename IndexType>                                     \
    GKO_DECLARE_RS_FILL_COARSE_ROWS_KERNEL(IndexType)


}  // namespace rs


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(rs, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_RS_KERNELS_HPP_
