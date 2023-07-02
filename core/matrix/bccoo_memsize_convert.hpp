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
#include "core/matrix/bccoo_helper.hpp"


namespace gko {
namespace matrix {
namespace bccoo {


/**
 *  Routines for mem_size computing
 */


/**
 *  Returns the size of the compressed data, which it is needed
 *  to store the data included in an individual compression object
 *  into an individual compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_ind_ind(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_individual_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;
    compr_idxs<IndexType> idxs_res;
    ValueType val_res;

    for (IndexType i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(start_rows_src, block_offsets_src, idxs_src);
        uint8 ind_src = get_position_newrow(compressed_data_src, idxs_src);
        get_next_position_value(compressed_data_src, ind_src, idxs_src,
                                val_src);
        get_detect_endblock(block_size_src, idxs_src);
        // Counting bytes to write (row,col,val) on result
        cnt_detect_newblock(idxs_src.row - idxs_res.row, idxs_res);
        IndexType col_src_res =
            cnt_position_newrow_mat_data(idxs_src.row, idxs_src.col, idxs_res);
        cnt_next_position_value(col_src_res, val_src, idxs_res);
        cnt_detect_endblock(block_size_res, idxs_res);
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the compressed data, which it is needed
 *  to store the data included in an individual compression object
 *  into a group compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_ind_grp(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_individual_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;

    compr_idxs<IndexType> idxs_res;

    for (IndexType i = 0; i < num_stored_elements; i += block_size_res) {
        IndexType block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_grp_idxs<IndexType> grp_idxs_res;
        grp_idxs_res.row_frst = idxs_src.row;
        grp_idxs_res.col_frst = idxs_src.col;
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(start_rows_src, block_offsets_src, idxs_src);
            uint8 ind_src = get_position_newrow(compressed_data_src, idxs_src);
            get_next_position_value(compressed_data_src, ind_src, idxs_src,
                                    val_src);
            get_detect_endblock(block_size_src, idxs_src);
            // Analyzing the impact of (row,col,val) in the block
            idxs_res.nblk = j;
            proc_group_keys<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                       grp_idxs_res);
        }
        // Counting bytes to write block on result
        cnt_group_keys<IndexType, ValueType>(block_size_local, grp_idxs_res,
                                             idxs_res);
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the compressed data, which it is needed
 *  to store the data included in a group compression object
 *  into an individual compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_grp_ind(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_group_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();
    const IndexType* start_cols_src = source->get_const_start_cols();
    const uint8* compression_types_src = source->get_const_compression_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;

    compr_idxs<IndexType> idxs_res;

    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local =
            std::min(block_size_src, num_stored_elements - i);
        compr_grp_idxs<IndexType> grp_idxs_src(
            start_rows_src, start_cols_src, block_size_local, idxs_src,
            compression_types_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_group_position_value<IndexType, ValueType>(
                compressed_data_src, grp_idxs_src, idxs_src, val_src);
            // Counting bytes to write (row,col,val) on result
            cnt_detect_newblock(idxs_src.row - idxs_res.row, idxs_res);
            IndexType col_src_res = cnt_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, idxs_res);
            cnt_next_position_value(col_src_res, val_src, idxs_res);
            cnt_detect_endblock(block_size_res, idxs_res);
        }
        idxs_src.blk++;
        idxs_src.shf = grp_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


/**
 *  Returns the size of the compressed data, which it is needed
 *  to store the data included in a group compression object
 *  into a group compression object whose block_size is specified
 */
template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_grp_grp(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    const IndexType block_size_res, size_type* mem_size)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_group_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();
    const IndexType* start_cols_src = source->get_const_start_cols();
    const uint8* compression_types_src = source->get_const_compression_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType val_src;

    const IndexType* start_rows_res = source->get_const_start_rows();
    const size_type* block_offsets_res = source->get_const_block_offsets();
    const uint8* compressed_data_res = source->get_const_compressed_data();
    const IndexType* start_cols_res = source->get_const_start_cols();
    const uint8* compression_types_res = source->get_const_compression_types();

    compr_idxs<IndexType> idxs_res;
    compr_grp_idxs<IndexType> grp_idxs_res;
    ValueType val_res;

    IndexType i_res = 0;
    IndexType block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        compr_grp_idxs<IndexType> grp_idxs_src(
            start_rows_src, start_cols_src, block_size_local_src, idxs_src,
            compression_types_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_group_position_value<IndexType, ValueType>(
                compressed_data_src, grp_idxs_src, idxs_src, val_src);
            proc_group_keys<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                       grp_idxs_res);
            idxs_res.nblk++;
            if (idxs_res.nblk == block_size_local_res) {
                // Counting bytes to write block on result
                cnt_group_keys<IndexType, ValueType>(block_size_local_res,
                                                     grp_idxs_res, idxs_res);
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                idxs_res.nblk = 0;
                grp_idxs_res = {};
            }
        }
        idxs_src.blk++;
        idxs_src.shf = grp_idxs_src.shf_val;
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
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->get_compression() == result->get_compression());

    // Try to remove static_cast
    if (source->get_num_stored_elements() > 0) {
        if (source->use_individual_compression()) {
            std::memcpy(result->get_start_rows(),
                        source->get_const_start_rows(),
                        source->get_num_blocks() * sizeof(IndexType));
            const size_type* block_offsets_src =
                source->get_const_block_offsets();
            size_type* block_offsets_res = result->get_block_offsets();
            std::memcpy(block_offsets_res, block_offsets_src,
                        (source->get_num_blocks() + 1) * sizeof(size_type));
            const uint8* compressed_data_src =
                source->get_const_compressed_data();
            uint8* compressed_data_res = result->get_compressed_data();
            std::memcpy(compressed_data_res, compressed_data_src,
                        source->get_num_bytes() * sizeof(uint8));
        } else {
            std::memcpy(result->get_start_rows(),
                        source->get_const_start_rows(),
                        source->get_num_blocks() * sizeof(IndexType));
            std::memcpy(result->get_start_cols(),
                        source->get_const_start_cols(),
                        source->get_num_blocks() * sizeof(IndexType));
            std::memcpy(result->get_compression_types(),
                        source->get_const_compression_types(),
                        source->get_num_blocks() * sizeof(uint8));
            std::memcpy(result->get_block_offsets(),
                        source->get_const_block_offsets(),
                        (source->get_num_blocks() + 1) * sizeof(size_type));
            std::memcpy(result->get_compressed_data(),
                        source->get_const_compressed_data(),
                        source->get_num_bytes() * sizeof(uint8));
        }
    }
}


/**
 *  This routine makes the conversion between two individual compression objects
 *  Additionally, finalize_op function is applied before to copy the values
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_ind_ind(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_individual_compression());
    GKO_ASSERT(result->use_individual_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* start_rows_res = result->get_start_rows();
    size_type* block_offsets_res = result->get_block_offsets();
    uint8* compressed_data_res = result->get_compressed_data();

    IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        block_offsets_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(start_rows_src, block_offsets_src, idxs_src);
        uint8 ind_src = get_position_newrow(compressed_data_src, idxs_src);
        get_next_position_value(compressed_data_src, ind_src, idxs_src,
                                val_src);
        get_detect_endblock(block_size_src, idxs_src);
        // Writing (row,col,val) to result
        val_res = finalize_op(val_src);
        put_detect_newblock(start_rows_res, idxs_src.row - idxs_res.row,
                            idxs_res);
        IndexType col_src_res = put_position_newrow_mat_data(
            idxs_src.row, idxs_src.col, compressed_data_res, idxs_res);
        put_next_position_value(compressed_data_res, col_src_res, val_res,
                                idxs_res);
        put_detect_endblock(block_offsets_res, block_size_res, idxs_res);
    }
    if (idxs_res.nblk > 0) {
        block_offsets_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between an individual compression object
 *  and a group compression object
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_ind_grp(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_individual_compression());
    GKO_ASSERT(result->use_group_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();

    const IndexType num_stored_elements = source->get_num_stored_elements();
    const IndexType block_size_src = source->get_block_size();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* start_rows_res = result->get_start_rows();
    size_type* block_offsets_res = result->get_block_offsets();
    uint8* compressed_data_res = result->get_compressed_data();
    IndexType* start_cols_res = result->get_start_cols();
    uint8* compression_types_res = result->get_compression_types();

    const IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    ValueType_res val_res;

    array<IndexType> rows_grp(exec, block_size_res);
    array<IndexType> cols_grp(exec, block_size_res);
    array<ValueType_res> vals_grp(exec, block_size_res);

    if (num_stored_elements > 0) {
        block_offsets_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i += block_size_res) {
        IndexType block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_grp_idxs<IndexType> grp_idxs_res;
        uint8 type_grp = {};

        grp_idxs_res.row_frst = idxs_src.row;
        grp_idxs_res.col_frst = idxs_src.col;
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(start_rows_src, block_offsets_src, idxs_src);
            uint8 ind_src = get_position_newrow(compressed_data_src, idxs_src);
            get_next_position_value(compressed_data_src, ind_src, idxs_src,
                                    val_src);
            get_detect_endblock(block_size_src, idxs_src);
            // Analyzing the impact of (row,col,val) in the block
            idxs_res.nblk = j;
            proc_group_keys<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                       grp_idxs_res);
            rows_grp.get_data()[j] = idxs_src.row;
            cols_grp.get_data()[j] = idxs_src.col;
            vals_grp.get_data()[j] = val_src;
        }
        // Writing block on result
        idxs_res.nblk = block_size_local;
        type_grp = write_compressed_data_grp_type(idxs_res, grp_idxs_res,
                                                  rows_grp, cols_grp, vals_grp,
                                                  compressed_data_res);
        start_rows_res[idxs_res.blk] = grp_idxs_res.row_frst;
        start_cols_res[idxs_res.blk] = grp_idxs_res.col_frst;
        compression_types_res[idxs_res.blk] = type_grp;
        block_offsets_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between a group compression object
 *  and an individual compression object
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_grp_ind(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_group_compression());
    GKO_ASSERT(result->use_individual_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();
    const IndexType* start_cols_src = source->get_const_start_cols();
    const uint8* compression_types_src = source->get_const_compression_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* start_rows_res = result->get_start_rows();
    size_type* block_offsets_res = result->get_block_offsets();
    uint8* compressed_data_res = result->get_compressed_data();
    IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        block_offsets_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local =
            std::min(block_size_src, num_stored_elements - i);

        compr_grp_idxs<IndexType> grp_idxs_src(
            start_rows_src, start_cols_src, block_size_local, idxs_src,
            compression_types_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_group_position_value<IndexType, ValueType_src>(
                compressed_data_src, grp_idxs_src, idxs_src, val_src);
            // Writing (row,col,val) to result
            val_res = val_src;
            put_detect_newblock(start_rows_res, idxs_src.row - idxs_res.row,
                                idxs_res);
            IndexType col_src_res = put_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, compressed_data_res, idxs_res);
            put_next_position_value(compressed_data_res, col_src_res, val_res,
                                    idxs_res);
            put_detect_endblock(block_offsets_res, block_size_res, idxs_res);
        }
        idxs_src.blk++;
        idxs_src.shf = grp_idxs_src.shf_val;
    }
    if (idxs_res.nblk > 0) {
        block_offsets_res[++idxs_res.blk] = idxs_res.shf;
    }
}


/**
 *  This routine makes the conversion between two group compression objects
 *  Additionally, finalize_op function is applied before to copy the values
 */
template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_grp_grp(
    std::shared_ptr<const Executor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    // This routine only is useful for master executor
    GKO_ASSERT(exec == exec->get_master());
    GKO_ASSERT(source->use_group_compression());
    GKO_ASSERT(result->use_group_compression());

    const IndexType* start_rows_src = source->get_const_start_rows();
    const size_type* block_offsets_src = source->get_const_block_offsets();
    const uint8* compressed_data_src = source->get_const_compressed_data();
    const IndexType* start_cols_src = source->get_const_start_cols();
    const uint8* compression_types_src = source->get_const_compression_types();

    const IndexType block_size_src = source->get_block_size();
    const size_type num_bytes_src = source->get_num_bytes();
    const IndexType num_stored_elements = source->get_num_stored_elements();

    compr_idxs<IndexType> idxs_src;
    ValueType_src val_src;

    IndexType* start_rows_res = result->get_start_rows();
    size_type* block_offsets_res = result->get_block_offsets();
    uint8* compressed_data_res = result->get_compressed_data();
    IndexType* start_cols_res = result->get_start_cols();
    uint8* compression_types_res = result->get_compression_types();

    const IndexType block_size_res = result->get_block_size();

    compr_idxs<IndexType> idxs_res;
    compr_grp_idxs<IndexType> grp_idxs_res;

    array<IndexType> rows_grp_res(exec, block_size_res);
    array<IndexType> cols_grp_res(exec, block_size_res);
    array<ValueType_res> vals_grp_res(exec, block_size_res);

    uint8 type_grp = {};
    IndexType i_res = 0;
    IndexType block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    grp_idxs_res.row_frst = idxs_src.row;
    grp_idxs_res.col_frst = idxs_src.col;
    if (num_stored_elements > 0) {
        block_offsets_res[0] = 0;
    }
    for (IndexType i = 0; i < num_stored_elements; i += block_size_src) {
        IndexType block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        compr_grp_idxs<IndexType> grp_idxs_src(
            start_rows_src, start_cols_src, block_size_local_src, idxs_src,
            compression_types_src[idxs_src.blk]);
        for (IndexType j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_group_position_value<IndexType, ValueType_src>(
                compressed_data_src, grp_idxs_src, idxs_src, val_src);
            // Analyzing the impact of (row,col,val) in the block
            proc_group_keys<IndexType>(idxs_src.row, idxs_src.col, idxs_res,
                                       grp_idxs_res);
            rows_grp_res.get_data()[idxs_res.nblk] = idxs_src.row;
            cols_grp_res.get_data()[idxs_res.nblk] = idxs_src.col;
            vals_grp_res.get_data()[idxs_res.nblk] =
                (ValueType_res)finalize_op(val_src);
            idxs_res.nblk++;
            if (idxs_res.nblk == block_size_local_res) {
                // Writing block on result
                idxs_res.nblk = block_size_local_res;
                type_grp = write_compressed_data_grp_type(
                    idxs_res, grp_idxs_res, rows_grp_res, cols_grp_res,
                    vals_grp_res, compressed_data_res);
                start_rows_res[idxs_res.blk] = grp_idxs_res.row_frst;
                start_cols_res[idxs_res.blk] = grp_idxs_res.col_frst;
                compression_types_res[idxs_res.blk] = type_grp;
                block_offsets_res[++idxs_res.blk] = idxs_res.shf;
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                idxs_res.nblk = 0;
                grp_idxs_res = {};
            }
        }
        idxs_src.blk++;
        idxs_src.shf = grp_idxs_src.shf_val;
    }
}


}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_MEMSIZE_CONVERT_HPP_
