/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/matrix/block_approx_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Block_Approx
 * @ingroup block_approx
 */
namespace block_approx {


template <typename IndexType>
void compute_block_ptrs(std::shared_ptr<const DefaultExecutor> exec,
                        const size_type num_blocks,
                        const size_type *block_sizes, IndexType *block_ptrs)
{
#pragma omp parallel for
    for (size_type b = 0; b < num_blocks; ++b) {
        block_ptrs[b] = block_sizes[b];
    }
    components::prefix_sum(exec, block_ptrs, num_blocks + 1);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_COMPUTE_BLOCK_PTRS_KERNEL);


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const DefaultExecutor> exec,
          const matrix::BlockApprox<matrix::Csr<ValueType, IndexType>> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    auto dense_b = const_cast<matrix::Dense<ValueType> *>(b);
    auto block_ptrs = a->get_block_ptrs();
    auto block_overlaps = a->get_overlaps().get_overlaps();
    auto overlap_unidir = a->get_overlaps().get_unidirectional_array();
    auto overlap_start = a->get_overlaps().get_overlap_at_start_array();
#pragma omp parallel for
    for (size_type i = 0; i < a->get_num_blocks(); ++i) {
        size_type offset = block_ptrs[i];
        auto loc_mtx = a->get_block_mtxs()[i];
        auto row_ptrs = loc_mtx->get_const_row_ptrs();
        auto col_idxs = loc_mtx->get_const_col_idxs();
        auto vals = loc_mtx->get_const_values();
        auto overlap_start_offset =
            (block_overlaps && (!overlap_unidir[i] || overlap_start[i]))
                ? block_overlaps[i]
                : 0;
        auto overlap_end_offset =
            (block_overlaps && (!overlap_unidir[i] || !overlap_start[i]))
                ? block_overlaps[i]
                : 0;
        auto loc_size = a->get_block_dimensions()[i] - overlap_start_offset -
                        overlap_end_offset;
        const auto loc_b = dense_b->create_submatrix(
            span{offset - overlap_start_offset,
                 offset + loc_size[0] + overlap_end_offset},
            span{0, dense_b->get_size()[1]});
        auto x_row_span = span{offset, offset + loc_size[0]};
        auto x_comp_span = span{0, a->get_block_dimensions()[i][0]};
        auto x_col_span = span{0, c->get_size()[1]};
        auto loc_x = c->create_submatrix(x_row_span, x_col_span);

        ValueType temp_val = 0.0;
        for (size_type row = 0; row < loc_size[0]; ++row) {
            for (size_type j = 0; j < loc_x->get_size()[1]; ++j) {
                if (x_comp_span.in_span(row)) {
                    loc_x->at(row, j) = zero<ValueType>();
                }
            }
            for (size_type j = 0; j < loc_x->get_size()[1]; ++j) {
                temp_val = loc_x->at(row, j);
                for (size_type k = row_ptrs[row];
                     k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
                    auto val = vals[k];
                    auto col = col_idxs[k];
                    temp_val += val * loc_b->at(col, j);
                    if (x_comp_span.in_span(row)) {
                        loc_x->at(row, j) = temp_val;
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Dense<ValueType> *alpha,
    const matrix::BlockApprox<matrix::Csr<ValueType, IndexType>> *a,
    const matrix::Dense<ValueType> *b, const matrix::Dense<ValueType> *beta,
    matrix::Dense<ValueType> *c)
{
    auto dense_b = const_cast<matrix::Dense<ValueType> *>(b);
    auto block_ptrs = a->get_block_ptrs();
#pragma omp parallel for
    for (size_type i = 0; i < a->get_num_blocks(); ++i) {
        auto loc_size = a->get_block_dimensions()[i];
        size_type offset = block_ptrs[i];
        auto loc_mtx = a->get_block_mtxs()[i];
        auto row_ptrs = loc_mtx->get_const_row_ptrs();
        auto col_idxs = loc_mtx->get_const_col_idxs();
        auto vals = loc_mtx->get_const_values();
        auto valpha = alpha->at(0, 0);
        auto vbeta = beta->at(0, 0);
        const auto loc_b =
            dense_b->create_submatrix(span{offset, offset + loc_size[0]},
                                      span{0, dense_b->get_size()[1]});
        auto loc_x = c->create_submatrix(span{offset, offset + loc_size[0]},
                                         span{0, c->get_size()[1]});

        for (size_type row = 0; row < loc_size[0]; ++row) {
            for (size_type j = 0; j < loc_x->get_size()[1]; ++j) {
                loc_x->at(row, j) *= vbeta;
            }
            for (size_type k = row_ptrs[row];
                 k < static_cast<size_type>(row_ptrs[row + 1]); ++k) {
                auto val = vals[k];
                auto col = col_idxs[k];
                for (size_type j = 0; j < loc_x->get_size()[1]; ++j) {
                    loc_x->at(row, j) += valpha * val * loc_b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_APPROX_ADVANCED_SPMV_KERNEL);


}  // namespace block_approx
}  // namespace omp
}  // namespace kernels
}  // namespace gko
