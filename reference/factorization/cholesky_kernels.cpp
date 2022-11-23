/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/factorization/cholesky_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Cholesky namespace.
 *
 * @ingroup factor
 */
namespace cholesky {


template <typename ValueType, typename IndexType>
void cholesky_symbolic_count(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* mtx,
    const factorization::elimination_forest<IndexType>& forest,
    IndexType* row_nnz, array<IndexType>&)
{
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto cols = mtx->get_const_col_idxs();
    const auto parent = forest.parents.get_const_data();
    vector<bool> visited(num_rows, {exec});
    for (IndexType row = 0; row < num_rows; row++) {
        IndexType count{};
        visited.assign(num_rows, false);
        visited[row] = true;
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        for (auto nz = row_begin; nz < row_end; nz++) {
            const auto col = cols[nz];
            if (col < row) {
                auto node = col;
                while (!visited[node]) {
                    visited[node] = true;
                    count++;
                    node = parent[node];
                }
            }
        }
        row_nnz[row] = count + 1;  // add diagonal entry
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT);


template <typename ValueType, typename IndexType>
void cholesky_symbolic_factorize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* mtx,
    const factorization::elimination_forest<IndexType>& forest,
    matrix::Csr<ValueType, IndexType>* l_factor, const array<IndexType>&)
{
    const auto num_rows = mtx->get_size()[0];
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto cols = mtx->get_const_col_idxs();
    const auto out_row_ptrs = l_factor->get_const_row_ptrs();
    const auto out_cols = l_factor->get_col_idxs();
    const auto parent = forest.parents.get_const_data();
    vector<bool> visited(num_rows, {exec});
    for (IndexType row = 0; row < num_rows; row++) {
        auto out_nz = out_row_ptrs[row];
        visited.assign(num_rows, false);
        visited[row] = true;
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        for (auto nz = row_begin; nz < row_end; nz++) {
            const auto col = cols[nz];
            if (col < row) {
                auto node = col;
                while (!visited[node]) {
                    visited[node] = true;
                    out_cols[out_nz++] = node;
                    node = parent[node];
                }
            }
        }
        out_cols[out_nz] = row;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE);


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                IndexType* transpose_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    lu_factorization::initialize(exec, mtx, factor_lookup_offsets,
                                 factor_lookup_descs, factor_lookup_storage,
                                 diag_idxs, factors);
    // convert to COO
    const auto nnz = factors->get_num_stored_elements();
    array<IndexType> row_idx_array{exec, nnz};
    const auto row_idxs = row_idx_array.get_data();
    const auto col_idxs = factors->get_const_col_idxs();
    components::convert_ptrs_to_idxs(exec, factors->get_const_row_ptrs(),
                                     factors->get_size()[0], row_idxs);
    // compute nonzero permutation for sparse transpose
    std::sort(transpose_idxs, transpose_idxs + nnz,
              [&](IndexType i, IndexType j) {
                  return std::tie(col_idxs[i], row_idxs[i]) <
                         std::tie(col_idxs[j], row_idxs[j]);
              });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_storage_offsets,
               const int64* lookup_descs, const int32* lookup_storage,
               const IndexType* diag_idxs, const IndexType* transpose_idxs,
               matrix::Csr<ValueType, IndexType>* factors,
               array<int>& tmp_storage)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FACTORIZE);


template <typename ValueType, typename IndexType>
void ldl_factorize(std::shared_ptr<const DefaultExecutor> exec,
                   const IndexType* lookup_storage_offsets,
                   const int64* lookup_descs, const int32* lookup_storage,
                   const IndexType* diag_idxs, const IndexType* transpose_idxs,
                   matrix::Csr<ValueType, IndexType>* factors,
                   array<int>& tmp_storage)
{}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LDL_FACTORIZE);


}  // namespace cholesky
}  // namespace reference
}  // namespace kernels
}  // namespace gko
