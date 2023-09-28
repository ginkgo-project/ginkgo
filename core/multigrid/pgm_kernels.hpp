/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#ifndef GKO_CORE_MULTIGRID_PGM_KERNELS_HPP_
#define GKO_CORE_MULTIGRID_PGM_KERNELS_HPP_


#include <ginkgo/core/multigrid/pgm.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {
namespace pgm {


#define GKO_DECLARE_PGM_MATCH_EDGE_KERNEL(IndexType)             \
    void match_edge(std::shared_ptr<const DefaultExecutor> exec, \
                    const array<IndexType>& strongest_neighbor,  \
                    array<IndexType>& agg)

#define GKO_DECLARE_PGM_COUNT_UNAGG_KERNEL(IndexType)             \
    void count_unagg(std::shared_ptr<const DefaultExecutor> exec, \
                     const array<IndexType>& agg, IndexType* num_unagg)

#define GKO_DECLARE_PGM_RENUMBER_KERNEL(IndexType)             \
    void renumber(std::shared_ptr<const DefaultExecutor> exec, \
                  array<IndexType>& agg, IndexType* num_agg)

#define GKO_DECLARE_PGM_SORT_AGG_KERNEL(IndexType)                            \
    void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num, \
                  IndexType* row_idxs, IndexType* col_idxs)

#define GKO_DECLARE_PGM_MAP_ROW_KERNEL(IndexType)                        \
    void map_row(std::shared_ptr<const DefaultExecutor> exec,            \
                 size_type num_fine_row, const IndexType* fine_row_ptrs, \
                 const IndexType* agg, IndexType* row_idxs)

#define GKO_DECLARE_PGM_MAP_COL_KERNEL(IndexType)                            \
    void map_col(std::shared_ptr<const DefaultExecutor> exec, size_type nnz, \
                 const IndexType* fine_col_idxs, const IndexType* agg,       \
                 IndexType* col_idxs)

#define GKO_DECLARE_PGM_COUNT_UNREPEATED_NNZ_KERNEL(IndexType)             \
    void count_unrepeated_nnz(std::shared_ptr<const DefaultExecutor> exec, \
                              size_type nnz, const IndexType* row_idxs,    \
                              const IndexType* col_idxs,                   \
                              size_type* coarse_nnz)

#define GKO_DECLARE_PGM_FIND_STRONGEST_NEIGHBOR(ValueType, IndexType)   \
    void find_strongest_neighbor(                                       \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* weight_mtx,            \
        const matrix::Diagonal<ValueType>* diag, array<IndexType>& agg, \
        array<IndexType>& strongest_neighbor)

#define GKO_DECLARE_PGM_ASSIGN_TO_EXIST_AGG(ValueType, IndexType)       \
    void assign_to_exist_agg(                                           \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::Csr<ValueType, IndexType>* weight_mtx,            \
        const matrix::Diagonal<ValueType>* diag, array<IndexType>& agg, \
        array<IndexType>& intermediate_agg)

#define GKO_DECLARE_PGM_SORT_ROW_MAJOR(ValueType, IndexType)         \
    void sort_row_major(std::shared_ptr<const DefaultExecutor> exec, \
                        size_type nnz, IndexType* row_idxs,          \
                        IndexType* col_idxs, ValueType* vals)

#define GKO_DECLARE_PGM_COMPUTE_COARSE_COO(ValueType, IndexType)              \
    void compute_coarse_coo(std::shared_ptr<const DefaultExecutor> exec,      \
                            size_type fine_nnz, const IndexType* row_idxs,    \
                            const IndexType* col_idxs, const ValueType* vals, \
                            matrix::Coo<ValueType, IndexType>* coarse_coo)

#define GKO_DECLARE_PGM_GATHER_INDEX(IndexType)                    \
    void gather_index(std::shared_ptr<const DefaultExecutor> exec, \
                      size_type num_res, const IndexType* orig,    \
                      const IndexType* gather_map, IndexType* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                               \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_MATCH_EDGE_KERNEL(IndexType);                  \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_COUNT_UNAGG_KERNEL(IndexType);                 \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_RENUMBER_KERNEL(IndexType);                    \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_SORT_AGG_KERNEL(IndexType);                    \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_MAP_ROW_KERNEL(IndexType);                     \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_MAP_COL_KERNEL(IndexType);                     \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_COUNT_UNREPEATED_NNZ_KERNEL(IndexType);        \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_PGM_FIND_STRONGEST_NEIGHBOR(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_PGM_ASSIGN_TO_EXIST_AGG(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_PGM_SORT_ROW_MAJOR(ValueType, IndexType);          \
    template <typename ValueType, typename IndexType>              \
    GKO_DECLARE_PGM_COMPUTE_COARSE_COO(ValueType, IndexType);      \
    template <typename IndexType>                                  \
    GKO_DECLARE_PGM_GATHER_INDEX(IndexType)


}  // namespace pgm


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(pgm, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MULTIGRID_PGM_KERNELS_HPP_
