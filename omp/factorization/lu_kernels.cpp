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

#include "core/factorization/lu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    const auto num_rows = mtx->get_size()[0];
    const auto mtx_row_ptrs = mtx->get_const_row_ptrs();
    const auto factor_row_ptrs = factors->get_const_row_ptrs();
    const auto mtx_cols = mtx->get_const_col_idxs();
    const auto factor_cols = factors->get_const_col_idxs();
    const auto mtx_vals = mtx->get_const_values();
    const auto factor_vals = factors->get_values();
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        const auto factor_begin = factor_row_ptrs[row];
        const auto factor_end = factor_row_ptrs[row + 1];
        const auto mtx_begin = mtx_row_ptrs[row];
        const auto mtx_end = mtx_row_ptrs[row + 1];
        std::fill(factor_vals + factor_begin, factor_vals + factor_end,
                  zero<ValueType>());
        matrix::csr::device_sparsity_lookup<IndexType> lookup{
            factor_row_ptrs,       factor_cols,         factor_lookup_offsets,
            factor_lookup_storage, factor_lookup_descs, row};
        for (auto nz = mtx_row_ptrs[row]; nz < mtx_row_ptrs[row + 1]; nz++) {
            const auto col = mtx_cols[nz];
            factor_vals[lookup.lookup_unsafe(col) + factor_begin] =
                mtx_vals[nz];
        }
        diag_idxs[row] = lookup.lookup_unsafe(row) + factor_begin;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               matrix::Csr<ValueType, IndexType>* factors,
               array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    const auto row_ptrs = factors->get_const_row_ptrs();
    const auto cols = factors->get_const_col_idxs();
    const auto vals = factors->get_values();
    vector<int> ready(num_rows, {exec});
#pragma omp parallel for schedule(monotonic : auto)
    for (size_type row = 0; row < num_rows; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_diag = diag_idxs[row];
        matrix::csr::device_sparsity_lookup<IndexType> lookup{
            row_ptrs, cols, lookup_offsets, lookup_storage, lookup_descs, row};
        for (auto lower_nz = row_begin; lower_nz < row_diag; lower_nz++) {
            const auto dep = cols[lower_nz];
            bool ready_flag{};
            while (!ready_flag) {
#pragma omp atomic read acquire
                ready_flag = ready[dep];
            }
            const auto dep_diag_idx = diag_idxs[dep];
            const auto dep_diag = vals[dep_diag_idx];
            const auto dep_end = row_ptrs[dep + 1];
            const auto scale = vals[lower_nz] / dep_diag;
            vals[lower_nz] = scale;
            for (auto dep_nz = dep_diag_idx + 1; dep_nz < dep_end; dep_nz++) {
                const auto col = cols[dep_nz];
                const auto val = vals[dep_nz];
                const auto nz = row_begin + lookup.lookup_unsafe(col);
                vals[nz] -= scale * val;
            }
        }
#pragma omp atomic write release
        ready[row] = true;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_FACTORIZE);


}  // namespace lu_factorization
}  // namespace omp
}  // namespace kernels
}  // namespace gko
