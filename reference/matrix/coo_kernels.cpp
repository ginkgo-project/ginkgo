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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/dense_kernels.hpp"


const int GKO_BCCOO_ROWS_MULTIPLE = 1;
const int GKO_BCCOO_COLS_8BITS = 2;
const int GKO_BCCOO_COLS_16BITS = 4;


namespace gko {
namespace kernels {
/**
 * @brief The Reference namespace.
 *
 * @ingroup reference
 */
namespace reference {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Coo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
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


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Coo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto coo_val = a->get_const_values();
    auto coo_col = a->get_const_col_idxs();
    auto coo_row = a->get_const_row_idxs();
    auto num_cols = b->get_size()[1];
    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
        for (size_type j = 0; j < num_cols; j++) {
            c->at(coo_row[i], j) += coo_val[i] * b->at(coo_col[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Coo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    auto coo_val = a->get_const_values();
    auto coo_col = a->get_const_col_idxs();
    auto coo_row = a->get_const_row_idxs();
    auto alpha_val = alpha->at(0, 0);
    auto num_cols = b->get_size()[1];
    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
        for (size_type j = 0; j < num_cols; j++) {
            c->at(coo_row[i], j) +=
                alpha_val * coo_val[i] * b->at(coo_col[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Coo<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    auto coo_val = source->get_const_values();
    auto coo_col = source->get_const_col_idxs();
    auto coo_row = source->get_const_row_idxs();
    for (size_type i = 0; i < source->get_num_stored_elements(); i++) {
        result->at(coo_row[i], coo_col[i]) += coo_val[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_FILL_IN_DENSE_KERNEL);


#define USE_BCCOO_STRUCT 1
template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Coo<ValueType, IndexType>* coo,
                    //                    IndexType* rows, IndexType* offsets,
                    //                    const size_type num_blocks,
                    const size_type block_size,
                    const matrix::bccoo::compression compress,
                    size_type* mem_size)
{
    if (compress == matrix::bccoo::compression::element) {
#ifdef USE_BCCOO_STRUCT
        // Computation of rows, offsets and m (mem_size)
        const IndexType* row_idxs = coo->get_const_row_idxs();
        const IndexType* col_idxs = coo->get_const_col_idxs();
        const ValueType* values = coo->get_const_values();
        const size_type num_rows = coo->get_size()[0];
        const size_type num_stored_elements = coo->get_num_stored_elements();
        compr_idxs idxs = {};
        for (size_type i = 0; i < num_stored_elements; i++) {
            const size_type row = row_idxs[i];
            const size_type col = col_idxs[i];
            const ValueType val = values[i];
            cnt_detect_newblock(idxs.nblk, idxs.shf, idxs.row, row - idxs.row,
                                idxs.col);
            size_type col_src_res = cnt_position_newrow_mat_data(
                row, col, idxs.shf, idxs.row, idxs.col);
            cnt_next_position_value(col_src_res, idxs.shf, idxs.col, val,
                                    idxs.nblk);
            cnt_detect_endblock(block_size, idxs.nblk, idxs.blk);
        }
        *mem_size = idxs.shf;
#else
        // Computation of rows, offsets and m (mem_size)
        const IndexType* row_idxs = coo->get_const_row_idxs();
        const IndexType* col_idxs = coo->get_const_col_idxs();
        const ValueType* values = coo->get_const_values();
        const size_type num_rows = coo->get_size()[0];
        const size_type num_stored_elements = coo->get_num_stored_elements();
        size_type nblk = 0;
        size_type blk = 0;
        size_type row_res = 0;
        size_type col_res = 0;
        size_type shf = 0;
        for (size_type i = 0; i < num_stored_elements; i++) {
            const size_type row = row_idxs[i];
            const size_type col = col_idxs[i];
            const ValueType val = values[i];
            cnt_detect_newblock(nblk, shf, row_res, row - row_res, col_res);
            size_type col_src_res =
                cnt_position_newrow_mat_data(row, col, shf, row_res, col_res);
            cnt_next_position_value(col_src_res, shf, col_res, val, nblk);
            cnt_detect_endblock(block_size, nblk, blk);
        }
        *mem_size = shf;
#endif
    } else {
        const IndexType* row_idxs = coo->get_const_row_idxs();
        const IndexType* col_idxs = coo->get_const_col_idxs();
        const ValueType* values = coo->get_const_values();
        auto num_rows = coo->get_size()[0];
        auto num_cols = coo->get_size()[1];
        auto num_stored_elements = coo->get_num_stored_elements();
        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        for (size_type i = 0; i < num_stored_elements; i++) {
            // std::cout << "COO_MEM: " << i << " - " << idxs.shf
            //					<< std::endl;
            const size_type row = row_idxs[i];
            const size_type col = col_idxs[i];
            const ValueType val = values[i];
            if (idxs.nblk == 0) {
                blk_idxs.row_frs = row;
                blk_idxs.col_frs = col;
            }
            blk_idxs.mul_row = blk_idxs.mul_row || (row != blk_idxs.row_frs);
            if (col < blk_idxs.col_frs) {
                blk_idxs.col_dif += (blk_idxs.col_frs - col);
                blk_idxs.col_frs = col;
            } else if (col > (blk_idxs.col_frs + blk_idxs.col_dif)) {
                blk_idxs.col_dif = col - blk_idxs.col_frs;
            }
            idxs.nblk++;
            if (idxs.nblk == block_size) {
                // Counting bytes to write block on result
                if (blk_idxs.mul_row) idxs.shf += block_size;
                if (blk_idxs.col_dif <= 0xFF) {
                    idxs.shf += block_size;
                } else if (blk_idxs.col_dif <= 0xFFFF) {
                    idxs.shf += 2 * block_size;
                } else {
                    idxs.shf += 4 * block_size;
                }
                idxs.shf += sizeof(ValueType) * block_size;
                idxs.blk++;
                idxs.nblk = 0;
                blk_idxs = {};
            }
        }
        if (idxs.nblk > 0) {
            // Counting bytes to write block on result
            if (blk_idxs.mul_row) idxs.shf += idxs.nblk;
            if (blk_idxs.col_dif <= 0xFF) {
                idxs.shf += idxs.nblk;
            } else if (blk_idxs.col_dif <= 0xFFFF) {
                idxs.shf += 2 * idxs.nblk;
            } else {
                idxs.shf += 4 * idxs.nblk;
            }
            idxs.shf += sizeof(ValueType) * idxs.nblk;
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
void convert_to_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Coo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
{
    if (result->use_element_compression()) {
        // std::cout << "COO -> ELEMENT" << std::endl;
#ifdef USE_BCCOO_STRUCT
        size_type block_size = result->get_block_size();
        IndexType* rows_data = result->get_rows();
        IndexType* offsets_data = result->get_offsets();
        uint8* chunk_data = result->get_chunk();

        // Computation of chunk
        const IndexType* row_idxs = source->get_const_row_idxs();
        const IndexType* col_idxs = source->get_const_col_idxs();
        const ValueType* values = source->get_const_values();
        const size_type num_rows = source->get_size()[0];
        const size_type num_stored_elements = source->get_num_stored_elements();
        compr_idxs idxs = {};

        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (size_type i = 0; i < num_stored_elements; i++) {
            const size_type row = row_idxs[i];
            const size_type col = col_idxs[i];
            const ValueType val = values[i];
            put_detect_newblock(chunk_data, rows_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, row - idxs.row, idxs.col);
            size_type col_src_res = put_position_newrow_mat_data(
                row, col, chunk_data, idxs.shf, idxs.row, idxs.col);
            put_next_position_value(chunk_data, idxs.nblk, col - idxs.col,
                                    idxs.shf, idxs.col, val);
            put_detect_endblock(offsets_data, idxs.shf, block_size, idxs.nblk,
                                idxs.blk);
        }
#else
        size_type block_size = result->get_block_size();
        IndexType* rows_data = result->get_rows();
        IndexType* offsets_data = result->get_offsets();
        uint8* chunk_data = result->get_chunk();

        // Computation of chunk
        const IndexType* row_idxs = source->get_const_row_idxs();
        const IndexType* col_idxs = source->get_const_col_idxs();
        const ValueType* values = source->get_const_values();
        const size_type num_rows = source->get_size()[0];
        const size_type num_stored_elements = source->get_num_stored_elements();
        size_type nblk = 0, blk = 0, row_res = 0, col_res = 0, shf = 0;
        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (size_type i = 0; i < num_stored_elements; i++) {
            const size_type row = row_idxs[i];
            const size_type col = col_idxs[i];
            const ValueType val = values[i];
            put_detect_newblock(chunk_data, rows_data, nblk, blk, shf, row_res,
                                row - row_res, col_res);
            size_type col_src_res = put_position_newrow_mat_data(
                row, col, chunk_data, shf, row_res, col_res);
            put_next_position_value(chunk_data, nblk, col - col_res, shf,
                                    col_res, val);
            put_detect_endblock(offsets_data, shf, block_size, nblk, blk);
        }
#endif
    } else {
        // std::cout << "COO -> BLOCK" << std::endl;
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

        compr_idxs idxs = {};
        compr_blk_idxs blk_idxs = {};
        uint8 type_blk = {};
        ValueType val;

        array<IndexType> rows_blk(exec, block_size);
        array<IndexType> cols_blk(exec, block_size);
        array<ValueType> vals_blk(exec, block_size);

        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (size_type i = 0; i < num_stored_elements; i++) {
            const size_type row = row_idxs[i];
            const size_type col = col_idxs[i];
            const ValueType val = values[i];
            if (idxs.nblk == 0) {
                blk_idxs.row_frs = row;
                blk_idxs.col_frs = col;
                blk_idxs.col_dif = 0;
            }
            rows_blk.get_data()[idxs.nblk] = row;
            cols_blk.get_data()[idxs.nblk] = col;
            vals_blk.get_data()[idxs.nblk] = val;
            blk_idxs.mul_row = blk_idxs.mul_row || (row != blk_idxs.row_frs);
            if (col < blk_idxs.col_frs) {
                blk_idxs.col_dif += (blk_idxs.col_frs - col);
                blk_idxs.col_frs = col;
            } else if (col > (blk_idxs.col_frs + blk_idxs.col_dif)) {
                blk_idxs.col_dif = col - blk_idxs.col_frs;
            }
            idxs.nblk++;
            if (idxs.nblk == block_size) {
                type_blk = {};
                if (blk_idxs.mul_row) {
                    for (size_type j = 0; j < block_size; j++) {
                        size_type row_src = rows_blk.get_data()[j];
                        // set_value_chunk<uint8>(chunk_data, shf+j,
                        set_value_chunk<uint8>(chunk_data, idxs.shf,
                                               row_src - blk_idxs.row_frs);
                        idxs.shf++;
                    }
                    type_blk |= GKO_BCCOO_ROWS_MULTIPLE;
                }
                if (blk_idxs.col_dif <= 0xFF) {
                    for (size_type j = 0; j < block_size; j++) {
                        uint8 col_dif =
                            cols_blk.get_data()[j] - blk_idxs.col_frs;
                        set_value_chunk<uint8>(chunk_data, idxs.shf, col_dif);
                        idxs.shf++;
                    }
                    type_blk |= GKO_BCCOO_COLS_8BITS;
                } else if (blk_idxs.col_dif <= 0xFFFF) {
                    for (size_type j = 0; j < block_size; j++) {
                        uint16 col_dif =
                            cols_blk.get_data()[j] - blk_idxs.col_frs;
                        set_value_chunk<uint16>(chunk_data, idxs.shf, col_dif);
                        idxs.shf += 2;
                    }
                    type_blk |= GKO_BCCOO_COLS_16BITS;
                } else {
                    for (size_type j = 0; j < block_size; j++) {
                        uint32 col_dif =
                            cols_blk.get_data()[j] - blk_idxs.col_frs;
                        set_value_chunk<uint32>(chunk_data, idxs.shf, col_dif);
                        idxs.shf += 4;
                    }
                }
                for (size_type j = 0; j < block_size; j++) {
                    ValueType val = vals_blk.get_data()[j];
                    set_value_chunk<ValueType>(chunk_data, idxs.shf, val);
                    idxs.shf += sizeof(ValueType);
                }
                rows_data[idxs.blk] = blk_idxs.row_frs;
                cols_data[idxs.blk] = blk_idxs.col_frs;
                types_data[idxs.blk] = type_blk;
                offsets_data[++idxs.blk] = idxs.shf;
                idxs.nblk = 0;
                type_blk = {};
                blk_idxs = {};
            }
        }
        if (idxs.nblk > 0) {
            type_blk = {};
            if (blk_idxs.mul_row) {
                for (size_type j = 0; j < idxs.nblk; j++) {
                    size_type row_src = rows_blk.get_data()[j];
                    // set_value_chunk<uint8>(chunk_data, shf+j,
                    set_value_chunk<uint8>(chunk_data, idxs.shf,
                                           row_src - blk_idxs.row_frs);
                    idxs.shf++;
                }
                type_blk |= GKO_BCCOO_ROWS_MULTIPLE;
            }
            if (blk_idxs.col_dif <= 0xFF) {
                for (size_type j = 0; j < idxs.nblk; j++) {
                    uint8 col_dif = cols_blk.get_data()[j] - blk_idxs.col_frs;
                    set_value_chunk<uint8>(chunk_data, idxs.shf, col_dif);
                    idxs.shf++;
                }
                type_blk |= GKO_BCCOO_COLS_8BITS;
            } else if (blk_idxs.col_dif <= 0xFFFF) {
                for (size_type j = 0; j < idxs.nblk; j++) {
                    uint16 col_dif = cols_blk.get_data()[j] - blk_idxs.col_frs;
                    set_value_chunk<uint16>(chunk_data, idxs.shf, col_dif);
                    idxs.shf += 2;
                }
                type_blk |= GKO_BCCOO_COLS_16BITS;
            } else {
                for (size_type j = 0; j < idxs.nblk; j++) {
                    uint32 col_dif = cols_blk.get_data()[j] - blk_idxs.col_frs;
                    set_value_chunk<uint32>(chunk_data, idxs.shf, col_dif);
                    idxs.shf += 4;
                }
            }
            for (size_type j = 0; j < idxs.nblk; j++) {
                val = vals_blk.get_data()[j];
                set_value_chunk<ValueType>(chunk_data, idxs.shf, val);
                idxs.shf += sizeof(ValueType);
            }
            rows_data[idxs.blk] = blk_idxs.row_frs;
            cols_data[idxs.blk] = blk_idxs.col_frs;
            types_data[idxs.blk] = type_blk;
            offsets_data[++idxs.blk] = idxs.shf;
            idxs.nblk = 0;
            type_blk = {};
            blk_idxs = {};
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Coo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto row_idxs = orig->get_const_row_idxs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const auto diag_size = diag->get_size()[0];
    const auto nnz = orig->get_num_stored_elements();
    auto diag_values = diag->get_values();

    for (size_type idx = 0; idx < nnz; idx++) {
        if (row_idxs[idx] == col_idxs[idx]) {
            diag_values[row_idxs[idx]] = values[idx];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_EXTRACT_DIAGONAL_KERNEL);


}  // namespace coo
}  // namespace reference
}  // namespace kernels
}  // namespace gko
