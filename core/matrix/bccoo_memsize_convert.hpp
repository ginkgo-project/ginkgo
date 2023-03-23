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

#ifndef GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_
#define GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


#include "core/base/unaligned_access.hpp"


namespace gko {
namespace matrix {
namespace bccoo {


/**
 *  Routines for mem_size computing
 */


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in an element compression object
 *  into an element compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_elm_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const size_type block_size_res, size_type* mem_size)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType val_src;
    compr_idxs idxs_res = {};
    ValueType val_res;

    for (size_type i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                            idxs_src.blk, idxs_src.shf, idxs_src.row,
                            idxs_src.col);
        uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                            idxs_src.row, idxs_src.col);
        get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                idxs_src.shf, idxs_src.col, val_src);
        get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
        // Counting bytes to write (row,col,val) on result
        cnt_detect_newblock(idxs_res.nblk, idxs_res.shf, idxs_res.row,
                            idxs_src.row - idxs_res.row, idxs_res.col);
        size_type col_src_res = cnt_position_newrow_mat_data(
            idxs_src.row, idxs_src.col, idxs_res.shf, idxs_res.row,
            idxs_res.col);
        cnt_next_position_value(col_src_res, idxs_res.shf, idxs_res.col,
                                val_src, idxs_res.nblk);
        cnt_detect_endblock(block_size_res, idxs_res.nblk, idxs_res.blk);
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in an element compression object
 *  into a block compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_elm_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const size_type block_size_res, size_type* mem_size)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType val_src;

    compr_idxs idxs_res = {};

    for (size_type i = 0; i < num_stored_elements; i += block_size_res) {
        size_type block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_blk_idxs blk_idxs_res = {};
        blk_idxs_res.row_frs = idxs_src.row;
        blk_idxs_res.col_frs = idxs_src.col;
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                                idxs_src.blk, idxs_src.shf, idxs_src.row,
                                idxs_src.col);
            uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                                idxs_src.row, idxs_src.col);
            get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                    idxs_src.shf, idxs_src.col, val_src);
            get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
            // Analyzing the impact of (row,col,val) in the block
            idxs_res.nblk = j;
            proc_block_indices(idxs_src.row, idxs_src.col, idxs_res,
                               blk_idxs_res);
        }
        // Counting bytes to write block on result
        cnt_block_indices<ValueType>(block_size_local, blk_idxs_res, idxs_res);
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in a blok compression object
 *  into an element compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_blk_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const size_type block_size_res, size_type* mem_size)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    auto block_size_src = source->get_block_size();
    auto num_bytes_src = source->get_num_bytes();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType val_src;

    compr_idxs idxs_res = {};

    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local =
            std::min(block_size_src, num_stored_elements - i);
        init_block_indices(rows_data_src, cols_data_src, block_size_local,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            // Counting bytes to write (row,col,val) on result
            cnt_detect_newblock(idxs_res.nblk, idxs_res.shf, idxs_res.row,
                                idxs_src.row - idxs_res.row, idxs_res.col);
            size_type col_src_res = cnt_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, idxs_res.shf, idxs_res.row,
                idxs_res.col);
            cnt_next_position_value(col_src_res, idxs_res.shf, idxs_res.col,
                                    val_src, idxs_res.nblk);
            cnt_detect_endblock(block_size_res, idxs_res.nblk, idxs_res.blk);
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the chunk, which it is need
 *  to store the data included in a block compression object
 *  into a block compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_blk_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const size_type block_size_res, size_type* mem_size)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    auto block_size_src = source->get_block_size();
    auto num_bytes_src = source->get_num_bytes();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType val_src;

    auto* rows_data_res = source->get_const_rows();
    auto* offsets_data_res = source->get_const_offsets();
    auto* chunk_data_res = source->get_const_chunk();
    auto* cols_data_res = source->get_const_cols();
    auto* types_data_res = source->get_const_types();

    compr_idxs idxs_res = {};
    compr_blk_idxs blk_idxs_res = {};
    ValueType val_res;

    size_type i_res = 0;
    size_type block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        init_block_indices(rows_data_src, cols_data_src, block_size_local_src,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            proc_block_indices(idxs_src.row, idxs_src.col, idxs_res,
                               blk_idxs_res);
            idxs_res.nblk++;
            if (idxs_res.nblk == block_size_local_res) {
                // Counting bytes to write block on result
                cnt_block_indices<ValueType>(block_size_local_res, blk_idxs_res,
                                             idxs_res);
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                idxs_res.nblk = 0;
                blk_idxs_res = {};
            }
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Routines for conversion between bccoo objects
 */


/**
 *  This routine makes a raw copy between bccoo objects whose block_size
 *  and compression are the same
 */
template <typename ValueType, typename IndexType>
void convert_to_bccoo_copy(std::shared_ptr<const Executor> exec,
                           const matrix::Bccoo<ValueType, IndexType>* source,
                           matrix::Bccoo<ValueType, IndexType>* result)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    // Try to remove static_cast
    if (source->get_num_stored_elements() > 0) {
        if (source->use_element_compression()) {
            std::memcpy((result->get_rows()), (source->get_const_rows()),
                        source->get_num_blocks() * sizeof(IndexType));
            auto offsets_data_src = source->get_const_offsets();
            auto offsets_data_res = result->get_offsets();
            std::memcpy((offsets_data_res), (offsets_data_src),
                        (source->get_num_blocks() + 1) * sizeof(size_type));
            auto chunk_data_src = source->get_const_chunk();
            auto chunk_data_res = result->get_chunk();
            std::memcpy((chunk_data_res), (chunk_data_src),
                        source->get_num_bytes() * sizeof(uint8));
        } else {
            std::memcpy((result->get_rows()), (source->get_const_rows()),
                        source->get_num_blocks() * sizeof(IndexType));
            std::memcpy((result->get_cols()), (source->get_const_cols()),
                        source->get_num_blocks() * sizeof(IndexType));
            std::memcpy((result->get_types()), (source->get_const_types()),
                        source->get_num_blocks() * sizeof(uint8));
            std::memcpy((result->get_offsets()), (source->get_const_offsets()),
                        (source->get_num_blocks() + 1) * sizeof(size_type));
            std::memcpy((result->get_chunk()), (source->get_const_chunk()),
                        source->get_num_bytes() * sizeof(uint8));
        }
    }
}


/**
 *  This routine makes the conversion between two element compression objects
 *  Additionally, finalize_op function is applied before to copy the values
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_elm_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();

    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                            idxs_src.blk, idxs_src.shf, idxs_src.row,
                            idxs_src.col);
        uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                            idxs_src.row, idxs_src.col);
        get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                idxs_src.shf, idxs_src.col, val_src);
        get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
        // Writing (row,col,val) to result
        val_res = finalize_op(val_src);
        put_detect_newblock(rows_data_res, idxs_res.nblk, idxs_res.blk,
                            idxs_res.row, idxs_src.row - idxs_res.row,
                            idxs_res.col);
        size_type col_src_res = put_position_newrow_mat_data(
            idxs_src.row, idxs_src.col, chunk_data_res, idxs_res.shf,
            idxs_res.row, idxs_res.col);
        put_next_position_value(chunk_data_res, idxs_res.nblk, col_src_res,
                                idxs_res.shf, idxs_res.col, val_res);
        put_detect_endblock(offsets_data_res, idxs_res.shf, block_size_res,
                            idxs_res.nblk, idxs_res.blk);
    }
    if (idxs_res.nblk > 0) {
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between an element compression object
 *  and a block compression object
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_elm_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    auto* cols_data_res = result->get_cols();
    auto* types_data_res = result->get_types();

    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    ValueType_res val_res;

    array<IndexType> rows_blk(exec, block_size_res);
    array<IndexType> cols_blk(exec, block_size_res);
    array<ValueType_res> vals_blk(exec, block_size_res);

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i += block_size_res) {
        size_type block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_blk_idxs blk_idxs_res = {};
        uint8 type_blk = {};

        blk_idxs_res.row_frs = idxs_src.row;
        blk_idxs_res.col_frs = idxs_src.col;
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                                idxs_src.blk, idxs_src.shf, idxs_src.row,
                                idxs_src.col);
            uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                                idxs_src.row, idxs_src.col);
            get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                    idxs_src.shf, idxs_src.col, val_src);
            get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
            // Analyzing the impact of (row,col,val) in the block
            idxs_res.nblk = j;
            proc_block_indices(idxs_src.row, idxs_src.col, idxs_res,
                               blk_idxs_res);
            rows_blk.get_data()[j] = idxs_src.row;
            cols_blk.get_data()[j] = idxs_src.col;
            vals_blk.get_data()[j] = val_src;
        }
        // Writing block on result
        idxs_res.nblk = block_size_local;
        type_blk = write_chunk_blk_type(idxs_res, blk_idxs_res, rows_blk,
                                        cols_blk, vals_blk, chunk_data_res);
        rows_data_res[idxs_res.blk] = blk_idxs_res.row_frs;
        cols_data_res[idxs_res.blk] = blk_idxs_res.col_frs;
        types_data_res[idxs_res.blk] = type_blk;
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between a block compression object
 *  and an element compression object
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_blk_elm(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    size_type block_size_src = source->get_block_size();
    size_type num_bytes_src = source->get_num_bytes();
    size_type num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local =
            std::min(block_size_src, num_stored_elements - i);

        init_block_indices(rows_data_src, cols_data_src, block_size_local,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType_src>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            // Writing (row,col,val) to result
            val_res = val_src;
            put_detect_newblock(rows_data_res, idxs_res.nblk, idxs_res.blk,
                                idxs_res.row, idxs_src.row - idxs_res.row,
                                idxs_res.col);
            size_type col_src_res = put_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, chunk_data_res, idxs_res.shf,
                idxs_res.row, idxs_res.col);
            put_next_position_value(chunk_data_res, idxs_res.nblk, col_src_res,
                                    idxs_res.shf, idxs_res.col, val_res);
            put_detect_endblock(offsets_data_res, idxs_res.shf, block_size_res,
                                idxs_res.nblk, idxs_res.blk);
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    if (idxs_res.nblk > 0) {
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between two block compression objects
 *  Additionally, finalize_op function is applied before to copy the values
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_blk_blk(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    // This routine only is usefel for master executor
    GKO_ASSERT(exec == exec->get_master());

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    auto block_size_src = source->get_block_size();
    auto num_bytes_src = source->get_num_bytes();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    auto* cols_data_res = result->get_cols();
    auto* types_data_res = result->get_types();

    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    compr_blk_idxs blk_idxs_res = {};

    array<IndexType> rows_blk_res(exec, block_size_res);
    array<IndexType> cols_blk_res(exec, block_size_res);
    array<ValueType_res> vals_blk_res(exec, block_size_res);

    uint8 type_blk = {};
    size_type i_res = 0;
    size_type block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    blk_idxs_res.row_frs = idxs_src.row;
    blk_idxs_res.col_frs = idxs_src.col;
    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        init_block_indices(rows_data_src, cols_data_src, block_size_local_src,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType_src>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            // Analyzing the impact of (row,col,val) in the block
            proc_block_indices(idxs_src.row, idxs_src.col, idxs_res,
                               blk_idxs_res);
            rows_blk_res.get_data()[idxs_res.nblk] = idxs_src.row;
            cols_blk_res.get_data()[idxs_res.nblk] = idxs_src.col;
            vals_blk_res.get_data()[idxs_res.nblk] =
                (ValueType_res)finalize_op(val_src);
            idxs_res.nblk++;
            if (idxs_res.nblk == block_size_local_res) {
                // Writing block on result
                idxs_res.nblk = block_size_local_res;
                type_blk = write_chunk_blk_type(idxs_res, blk_idxs_res,
                                                rows_blk_res, cols_blk_res,
                                                vals_blk_res, chunk_data_res);
                rows_data_res[idxs_res.blk] = blk_idxs_res.row_frs;
                cols_data_res[idxs_res.blk] = blk_idxs_res.col_frs;
                types_data_res[idxs_res.blk] = type_blk;
                offsets_data_res[++idxs_res.blk] = idxs_res.shf;
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                idxs_res.nblk = 0;
                blk_idxs_res = {};
            }
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
}


}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_
