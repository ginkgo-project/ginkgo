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

#include "core/matrix/coo_kernels.hpp"


#include <array>


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "omp/components/atomic.hpp"


namespace gko {
namespace kernels {
/**
 * @brief OpenMP namespace.
 *
 * @ingroup omp
 */
namespace omp {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Coo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Coo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <int block_size, typename ValueType, typename IndexType>
void spmv2_blocked(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Coo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   matrix::Dense<ValueType>* c, ValueType scale)
{
    GKO_ASSERT(b->get_size()[1] > block_size);
    const auto coo_val = a->get_const_values();
    const auto coo_col = a->get_const_col_idxs();
    const auto coo_row = a->get_const_row_idxs();
    const auto num_rhs = b->get_size()[1];
    const auto rounded_rhs = num_rhs / block_size * block_size;
    const auto sentinel_row = a->get_size()[0] + 1;
    const auto nnz = a->get_num_stored_elements();

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<size_type>(ceildiv(nnz, num_threads));
        const auto thread_id = static_cast<size_type>(omp_get_thread_num());
        const auto begin = work_per_thread * thread_id;
        const auto end = std::min(begin + work_per_thread, nnz);
        if (begin < end) {
            const auto first = begin > 0 ? coo_row[begin - 1] : sentinel_row;
            const auto last = end < nnz ? coo_row[end] : sentinel_row;
            auto nz = begin;
            std::array<ValueType, block_size> partial_sum;
            if (first != sentinel_row) {
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
                    // handle row overlap with previous thread: block partial
                    // sums
                    partial_sum.fill(zero<ValueType>());
                    for (auto local_nz = nz;
                         local_nz < end && coo_row[local_nz] == first;
                         local_nz++) {
                        const auto col = coo_col[local_nz];
#pragma unroll
                        for (size_type i = 0; i < block_size; i++) {
                            const auto rhs = i + rhs_base;
                            partial_sum[i] +=
                                scale * coo_val[local_nz] * b->at(col, rhs);
                        }
                    }
                    // handle row overlap with previous thread: block add to
                    // memory
#pragma unroll
                    for (size_type i = 0; i < block_size; i++) {
                        const auto rhs = i + rhs_base;
                        atomic_add(c->at(first, rhs), partial_sum[i]);
                    }
                }
                // handle row overlap with previous thread: remainder partial
                // sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end && coo_row[nz] == first; nz++) {
                    const auto row = first;
                    const auto col = coo_col[nz];
                    for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                        partial_sum[rhs - rounded_rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with previous thread: remainder add to
                // memory
                for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(first, rhs),
                               partial_sum[rhs - rounded_rhs]);
                }
            }
            // handle non-overlapping rows
            for (; nz < end && coo_row[nz] != last; nz++) {
                const auto row = coo_row[nz];
                const auto col = coo_col[nz];
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
#pragma unroll
                    for (size_type i = 0; i < block_size; i++) {
                        const auto rhs = i + rhs_base;
                        c->at(row, rhs) +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                    c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
                }
            }
            if (last != sentinel_row) {
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
                    // handle row overlap with following thread: block partial
                    // sums
                    partial_sum.fill(zero<ValueType>());
                    for (auto local_nz = nz; local_nz < end; local_nz++) {
                        const auto col = coo_col[local_nz];
#pragma unroll
                        for (size_type i = 0; i < block_size; i++) {
                            const auto rhs = i + rhs_base;
                            partial_sum[i] +=
                                scale * coo_val[local_nz] * b->at(col, rhs);
                        }
                    }
                    // handle row overlap with following thread: block add to
                    // memory
#pragma unroll
                    for (size_type i = 0; i < block_size; i++) {
                        const auto rhs = i + rhs_base;
                        const auto row = last;
                        atomic_add(c->at(row, rhs), partial_sum[i]);
                    }
                }
                // handle row overlap with following thread: block partial sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end; nz++) {
                    const auto col = coo_col[nz];
                    for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                        partial_sum[rhs - rounded_rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with following thread: block add to memory
                for (size_type rhs = rounded_rhs; rhs < num_rhs; rhs++) {
                    const auto row = last;
                    atomic_add(c->at(row, rhs), partial_sum[rhs - rounded_rhs]);
                }
            }
        }
    }
}


template <int num_rhs, typename ValueType, typename IndexType>
void spmv2_small_rhs(std::shared_ptr<const OmpExecutor> exec,
                     const matrix::Coo<ValueType, IndexType>* a,
                     const matrix::Dense<ValueType>* b,
                     matrix::Dense<ValueType>* c, ValueType scale)
{
    GKO_ASSERT(b->get_size()[1] == num_rhs);
    const auto coo_val = a->get_const_values();
    const auto coo_col = a->get_const_col_idxs();
    const auto coo_row = a->get_const_row_idxs();
    const auto sentinel_row = a->get_size()[0] + 1;
    const auto nnz = a->get_num_stored_elements();

#pragma omp parallel
    {
        const auto num_threads = omp_get_num_threads();
        const auto work_per_thread =
            static_cast<size_type>(ceildiv(nnz, num_threads));
        const auto thread_id = static_cast<size_type>(omp_get_thread_num());
        const auto begin = work_per_thread * thread_id;
        const auto end = std::min(begin + work_per_thread, nnz);
        if (begin < end) {
            const auto first = begin > 0 ? coo_row[begin - 1] : sentinel_row;
            const auto last = end < nnz ? coo_row[end] : sentinel_row;
            auto nz = begin;
            std::array<ValueType, num_rhs> partial_sum;
            if (first != sentinel_row) {
                // handle row overlap with previous thread: partial sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end && coo_row[nz] == first; nz++) {
                    const auto col = coo_col[nz];
#pragma unroll
                    for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                        partial_sum[rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with previous thread: add to memory
#pragma unroll
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    atomic_add(c->at(first, rhs), partial_sum[rhs]);
                }
            }
            // handle non-overlapping rows
            for (; nz < end && coo_row[nz] != last; nz++) {
                const auto row = coo_row[nz];
                const auto col = coo_col[nz];
#pragma unroll
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    c->at(row, rhs) += scale * coo_val[nz] * b->at(col, rhs);
                }
            }
            if (last != sentinel_row) {
                // handle row overlap with following thread: partial sums
                partial_sum.fill(zero<ValueType>());
                for (; nz < end; nz++) {
                    const auto col = coo_col[nz];
#pragma unroll
                    for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                        partial_sum[rhs] +=
                            scale * coo_val[nz] * b->at(col, rhs);
                    }
                }
                // handle row overlap with following thread: add to memory
#pragma unroll
                for (size_type rhs = 0; rhs < num_rhs; rhs++) {
                    const auto row = last;
                    atomic_add(c->at(row, rhs), partial_sum[rhs]);
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void generic_spmv2(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Coo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   matrix::Dense<ValueType>* c, ValueType scale)
{
    const auto num_rhs = b->get_size()[1];
    if (num_rhs <= 0) {
        return;
    }
    if (num_rhs == 1) {
        spmv2_small_rhs<1>(exec, a, b, c, scale);
        return;
    }
    if (num_rhs == 2) {
        spmv2_small_rhs<2>(exec, a, b, c, scale);
        return;
    }
    if (num_rhs == 3) {
        spmv2_small_rhs<3>(exec, a, b, c, scale);
        return;
    }
    if (num_rhs == 4) {
        spmv2_small_rhs<4>(exec, a, b, c, scale);
        return;
    }
    spmv2_blocked<4>(exec, a, b, c, scale);
}


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Coo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    generic_spmv2(exec, a, b, c, one<ValueType>());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Coo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    generic_spmv2(exec, a, b, c, alpha->at(0, 0));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Coo<ValueType, IndexType>* coo,
                    const IndexType block_size,
                    const matrix::bccoo::compression compress,
                    size_type* mem_size)
{
    if (compress == matrix::bccoo::compression::element) {
        // For element compression objects
        const IndexType* row_idxs = coo->get_const_row_idxs();
        const IndexType* col_idxs = coo->get_const_col_idxs();
        const ValueType* values = coo->get_const_values();
        const IndexType num_rows = coo->get_size()[0];
        const IndexType num_stored_elements = coo->get_num_stored_elements();
        matrix::bccoo::compr_idxs<IndexType> idxs;
        for (IndexType i = 0; i < num_stored_elements; i++) {
            const IndexType row = row_idxs[i];
            const IndexType col = col_idxs[i];
            const ValueType val = values[i];
            // Counting bytes to write (row,col,val) on result
            matrix::bccoo::cnt_detect_newblock<IndexType>(row - idxs.row, idxs);
            IndexType col_src_res =
                matrix::bccoo::cnt_position_newrow_mat_data(row, col, idxs);
            matrix::bccoo::cnt_next_position_value(col_src_res, val, idxs);
            matrix::bccoo::cnt_detect_endblock(block_size, idxs);
        }
        *mem_size = idxs.shf;
    } else {
        // For block compression objects
        const IndexType* row_idxs = coo->get_const_row_idxs();
        const IndexType* col_idxs = coo->get_const_col_idxs();
        const ValueType* values = coo->get_const_values();
        auto num_rows = coo->get_size()[0];
        auto num_cols = coo->get_size()[1];
        auto num_stored_elements = coo->get_num_stored_elements();
        matrix::bccoo::compr_idxs<IndexType> idxs;
        matrix::bccoo::compr_blk_idxs<IndexType> blk_idxs;
        for (IndexType i = 0; i < num_stored_elements; i++) {
            const IndexType row = row_idxs[i];
            const IndexType col = col_idxs[i];
            const ValueType val = values[i];
            // Counting bytes to write block on result
            matrix::bccoo::cnt_block_indices<IndexType, ValueType>(
                block_size, blk_idxs, idxs);
            idxs.nblk++;
            if (idxs.nblk == block_size) {
                // Counting bytes to write block on result
                matrix::bccoo::cnt_block_indices<IndexType, ValueType>(
                    block_size, blk_idxs, idxs);
                idxs.blk++;
                idxs.nblk = 0;
                blk_idxs = {};
            }
        }
        if (idxs.nblk > 0) {
            // Counting bytes to write block on result
            matrix::bccoo::cnt_block_indices<IndexType, ValueType>(
                block_size, blk_idxs, idxs);
            idxs.blk++;
            idxs.nblk = 0;
            blk_idxs = {};
        }
        *mem_size = idxs.shf;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Coo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
{
    if (result->use_element_compression()) {
        // For element compression objects
        IndexType block_size = result->get_block_size();
        IndexType* rows_data = result->get_rows();
        size_type* offsets_data = result->get_offsets();
        uint8* chunk_data = result->get_chunk();

        // Computation of chunk
        const IndexType* row_idxs = source->get_const_row_idxs();
        const IndexType* col_idxs = source->get_const_col_idxs();
        const ValueType* values = source->get_const_values();
        const IndexType num_rows = source->get_size()[0];
        const IndexType num_stored_elements = source->get_num_stored_elements();
        matrix::bccoo::compr_idxs<IndexType> idxs;

        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (IndexType i = 0; i < num_stored_elements; i++) {
            const IndexType row = row_idxs[i];
            const IndexType col = col_idxs[i];
            const ValueType val = values[i];
            // Writing (row,col,val) to result
            matrix::bccoo::put_detect_newblock(chunk_data, rows_data,
                                               row - idxs.row, idxs);
            IndexType col_src_res = matrix::bccoo::put_position_newrow_mat_data(
                row, col, chunk_data, idxs);
            matrix::bccoo::put_next_position_value(chunk_data, col - idxs.col,
                                                   val, idxs);
            matrix::bccoo::put_detect_endblock(offsets_data, block_size, idxs);
        }
        if (idxs.nblk > 0) {
            offsets_data[idxs.blk + 1] = idxs.shf;
        }
    } else {
        // For block compression objects
        const IndexType* row_idxs = source->get_const_row_idxs();
        const IndexType* col_idxs = source->get_const_col_idxs();
        const ValueType* values = source->get_const_values();
        auto num_rows = source->get_size()[0];
        auto num_cols = source->get_size()[1];

        auto* rows_data = result->get_rows();
        auto* cols_data = result->get_cols();
        auto* types_data = result->get_types();
        auto* offsets_data = result->get_offsets();
        auto* chunk_data = result->get_chunk();

        auto num_stored_elements = result->get_num_stored_elements();
        auto block_size = result->get_block_size();

        matrix::bccoo::compr_idxs<IndexType> idxs;
        matrix::bccoo::compr_blk_idxs<IndexType> blk_idxs;
        uint8 type_blk = {};
        ValueType val;

        array<IndexType> rows_blk(exec, block_size);
        array<IndexType> cols_blk(exec, block_size);
        array<ValueType> vals_blk(exec, block_size);

        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (IndexType i = 0; i < num_stored_elements; i++) {
            const IndexType row = row_idxs[i];
            const IndexType col = col_idxs[i];
            const ValueType val = values[i];
            // Analyzing the impact of (row,col,val) in the block
            matrix::bccoo::proc_block_indices<IndexType>(row, col, idxs,
                                                         blk_idxs);
            rows_blk.get_data()[idxs.nblk] = row;
            cols_blk.get_data()[idxs.nblk] = col;
            vals_blk.get_data()[idxs.nblk] = val;
            idxs.nblk++;
            if (idxs.nblk == block_size) {
                // Writing block on result
                type_blk = matrix::bccoo::write_chunk_blk_type(
                    idxs, blk_idxs, rows_blk, cols_blk, vals_blk, chunk_data);
                rows_data[idxs.blk] = blk_idxs.row_frst;
                cols_data[idxs.blk] = blk_idxs.col_frst;
                types_data[idxs.blk] = type_blk;
                offsets_data[++idxs.blk] = idxs.shf;
                idxs.nblk = 0;
                blk_idxs = {};
            }
        }
        if (idxs.nblk > 0) {
            // Writing block on result
            type_blk = matrix::bccoo::write_chunk_blk_type(
                idxs, blk_idxs, rows_blk, cols_blk, vals_blk, chunk_data);
            rows_data[idxs.blk] = blk_idxs.row_frst;
            cols_data[idxs.blk] = blk_idxs.col_frst;
            types_data[idxs.blk] = type_blk;
            offsets_data[++idxs.blk] = idxs.shf;
            idxs.nblk = 0;
            blk_idxs = {};
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_BCCOO_KERNEL);


}  // namespace coo
}  // namespace omp
}  // namespace kernels
}  // namespace gko
