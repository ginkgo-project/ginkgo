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

#ifndef GKO_CORE_MATRIX_BCCOO_HELPER_HPP_
#define GKO_CORE_MATRIX_BCCOO_HELPER_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>


#include "core/base/unaligned_access.hpp"
#include "core/matrix/bccoo_aux_structs.hpp"


namespace gko {
namespace matrix {
namespace bccoo {


/*
 *  Constants to manage bccoo objects
 */

constexpr uint8 cst_mark_end_row = 0xFF;
constexpr uint8 cst_mark_size_big_row = 0xFE;
constexpr uint8 cst_mark_size_medium_row = 0xFD;

constexpr uint8 cst_max_size_small_idxs_row = 0xFF;
constexpr uint8 cst_max_col_diff_small = 0xFC;
constexpr uint16 cst_max_col_diff_medium = 0xFFFF;
constexpr uint32 cst_max_col_diff_large = 0xFFFFFFFF;


/*
 *  Methods for managing bccoo objects
 */


// Adapts idxs assuming that col_src_res was added
template <typename IndexType>
inline void cnt_next_position(const IndexType col_src_res,
                              compr_idxs<IndexType>& idxs)

{
    if (col_src_res <= cst_max_col_diff_small) {
        idxs.shf++;
    } else if (col_src_res <= cst_max_col_diff_medium) {
        idxs.shf += sizeof(uint16) + 1;
    } else if (col_src_res <= cst_max_col_diff_large) {
        idxs.shf += sizeof(uint32) + 1;
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    idxs.col += col_src_res;
}


// Adapts idxs assuming that col_src_res and val were added
template <typename IndexType, typename ValueType>
inline void cnt_next_position_value(const IndexType col_src_res,
                                    const ValueType val,
                                    compr_idxs<IndexType>& idxs)
{
    cnt_next_position(col_src_res, idxs);
    idxs.shf += sizeof(ValueType);
    idxs.nblk++;
}


// From compressed_data and key, adapts idxs and gets idxs.col
template <typename IndexType>
inline void get_next_position(const uint8* compressed_data, const uint8 key,
                              compr_idxs<IndexType>& idxs)
{
    if (key < cst_mark_size_medium_row) {
        idxs.col += key;
        idxs.shf++;
    } else if (key == cst_mark_size_medium_row) {
        idxs.shf++;
        idxs.col += get_value_compressed_data_and_increment<uint16>(
            compressed_data, idxs.shf);
    } else {
        idxs.shf++;
        idxs.col += get_value_compressed_data_and_increment<uint32>(
            compressed_data, idxs.shf);
    }
}


// From compressed_data and key, adapts idxs and gets idxs.col and val
template <typename IndexType, typename ValueType>
inline void get_next_position_value(const uint8* compressed_data,
                                    const uint8 key,
                                    compr_idxs<IndexType>& idxs, ValueType& val)
{
    get_next_position(compressed_data, key, idxs);
    val = get_value_compressed_data_and_increment<ValueType>(compressed_data,
                                                             idxs.shf);
    idxs.nblk++;
}


// From compressed_data and key, adapts idxs and gets idxs.col and val
// Then applies finalize_op on val, writing it on compressed_data and
// returning it
template <typename IndexType, typename ValueType, typename Callable>
inline void get_next_position_value_put(uint8* compressed_data, const uint8 key,
                                        compr_idxs<IndexType>& idxs,
                                        ValueType& val, Callable finalize_op)
{
    get_next_position(compressed_data, key, idxs);
    val = get_value_compressed_data<ValueType>(compressed_data, idxs.shf);
    val = finalize_op(val);
    set_value_compressed_data_and_increment<ValueType>(compressed_data,
                                                       idxs.shf, val);
    idxs.nblk++;
}


// Writes col_src_res on compressed_data, and update idxs
template <typename IndexType>
inline void put_next_position(uint8* compressed_data,
                              const IndexType col_src_res,
                              compr_idxs<IndexType>& idxs)
{
    if (col_src_res <= cst_max_col_diff_small) {
        set_value_compressed_data_and_increment<uint8>(compressed_data,
                                                       idxs.shf, col_src_res);
        idxs.col += col_src_res;
    } else if (col_src_res <= cst_max_col_diff_medium) {
        set_value_compressed_data_and_increment<uint8>(
            compressed_data, idxs.shf, cst_mark_size_medium_row);
        set_value_compressed_data_and_increment<uint16>(compressed_data,
                                                        idxs.shf, col_src_res);
        idxs.col += col_src_res;
    } else if (col_src_res <= cst_max_col_diff_large) {
        set_value_compressed_data_and_increment<uint8>(
            compressed_data, idxs.shf, cst_mark_size_big_row);
        set_value_compressed_data_and_increment<uint32>(compressed_data,
                                                        idxs.shf, col_src_res);
        idxs.col += col_src_res;
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}


// Writes col_src_res and val on compressed_data, and update idxs
template <typename IndexType, typename ValueType>
inline void put_next_position_value(uint8* compressed_data,
                                    const IndexType col_src_res,
                                    const ValueType val,
                                    compr_idxs<IndexType>& idxs)
{
    put_next_position(compressed_data, col_src_res, idxs);
    set_value_compressed_data_and_increment<ValueType>(compressed_data,
                                                       idxs.shf, val);
    idxs.nblk++;
}


// Detects if a new block appearing when a Bccoo is read, adapting idxs
template <typename IndexType>
inline void get_detect_newblock(const IndexType* start_rows,
                                const size_type* block_offsets,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == 0) {
        idxs.row = start_rows[idxs.blk];
        idxs.col = 0;
        idxs.shf = block_offsets[idxs.blk];
    }
}


// Detects if a new block appearing when a Bccoo is written, adapting idxs
// If true, start_rows is updated
template <typename IndexType>
inline void put_detect_newblock(IndexType* start_rows,
                                const IndexType row_src_res,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == 0) {
        idxs.row += row_src_res;
        idxs.col = 0;
        start_rows[idxs.blk] = idxs.row;
    }
}


// Detects if a new block appearing when a Bccoo is written, adapting idxs
// If true, start_rows is updated
// If a new row within a block is detected, compressed_data is updated
template <typename IndexType>
inline void put_detect_newblock(uint8* compressed_data, IndexType* start_rows,
                                const size_type row_src_res,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == 0) {
        idxs.row += row_src_res;
        idxs.col = 0;
        start_rows[idxs.blk] = idxs.row;
    } else if (row_src_res != 0) {  // new row
        idxs.row += row_src_res;
        idxs.col = 0;
        for (size_type i = 0; i < row_src_res; i++) {
            set_value_compressed_data_and_increment<uint8>(
                compressed_data, idxs.shf, cst_mark_end_row);
        }
    }
}


// Detects if a new block appearing when a Bccoo is written, adapting idxs
// Both new block and new row within a block are considered
template <typename IndexType>
inline void cnt_detect_newblock(const IndexType row_src_res,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == 0) {
        idxs.row += row_src_res;
        idxs.col = 0;
    } else if (row_src_res != 0) {  // new row
        idxs.row += row_src_res;
        idxs.col = 0;
        idxs.shf += row_src_res;
    }
}


// Detects if a new block appearing when a Bccoo is read, adapting idxs
// If true and a new row is also detected, rows_ptrs is updated
template <typename IndexType>
inline void get_detect_newblock_csr(const IndexType* start_rows,
                                    const size_type* block_offsets,
                                    IndexType* row_ptrs, IndexType pos,
                                    compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == 0) {
        if (idxs.row != start_rows[idxs.blk]) {
            idxs.row = start_rows[idxs.blk];
            row_ptrs[idxs.row] = pos;
        }
        idxs.col = 0;
        idxs.shf = block_offsets[idxs.blk];
    }
}


// Adapts idxs, assuming (row,col) position is added and returning the column
// difference
template <typename IndexType>
inline IndexType cnt_position_newrow_mat_data(const IndexType row_mat_data,
                                              const IndexType col_mat_data,
                                              compr_idxs<IndexType>& idxs)
{
    if (row_mat_data != idxs.row) {
        idxs.shf += row_mat_data - idxs.row;
        idxs.row = row_mat_data;
        idxs.col = 0;
    }
    return (col_mat_data - idxs.col);
}


// Detects the position of the next position, updating idxs and returning key
template <typename IndexType>
inline uint8 get_position_newrow(const uint8* compressed_data,
                                 compr_idxs<IndexType>& idxs)
{
    uint8 key = get_value_compressed_data<uint8>(compressed_data, idxs.shf);
    while (key == cst_mark_end_row) {
        idxs.row++;
        idxs.shf++;
        idxs.col = 0;
        key = get_value_compressed_data<uint8>(compressed_data, idxs.shf);
    }
    return key;
}


// Detects the position of the next position, updating idxs and returning key
// Also writes row_ptrs per each new row
template <typename IndexType>
inline uint8 get_position_newrow_csr(const uint8* compressed_data,
                                     IndexType* row_ptrs, IndexType pos,
                                     compr_idxs<IndexType>& idxs)
{
    uint8 key = get_value_compressed_data<uint8>(compressed_data, idxs.shf);
    while (key == cst_mark_end_row) {
        idxs.row++;
        idxs.col = 0;
        row_ptrs[idxs.row] = pos;
        idxs.shf++;
        key = get_value_compressed_data<uint8>(compressed_data, idxs.shf);
    }
    return key;
}


// Detects the position of the next position, updating idxs_src and returning
// key_src Also writes in compressed_data_res and start_rows_res, updating
// idxs_res
template <typename IndexType>
inline uint8 get_position_newrow_put(const uint8* compressed_data_src,
                                     compr_idxs<IndexType>& idxs_src,
                                     uint8* compressed_data_res,
                                     IndexType* start_rows_res,
                                     compr_idxs<IndexType>& idxs_res)
{
    uint8 key_src =
        get_value_compressed_data<uint8>(compressed_data_src, idxs_src.shf);
    while (key_src == cst_mark_end_row) {
        idxs_src.row++;
        idxs_src.col = 0;
        idxs_src.shf++;
        key_src =
            get_value_compressed_data<uint8>(compressed_data_src, idxs_src.shf);
        idxs_res.row++;
        idxs_res.col = 0;
        if (idxs_res.nblk == 0) {
            start_rows_res[idxs_res.blk] = idxs_res.row;
        } else {
            set_value_compressed_data_and_increment<uint8>(
                compressed_data_res, idxs_res.shf, cst_mark_end_row);
        }
    }
    return key_src;
}


// Writes compressed_data, assuming (row,col) position is added, updating idxs
// and returning the column differenc
template <typename IndexType>
inline size_type put_position_newrow_mat_data(const IndexType row_mat_data,
                                              const IndexType col_mat_data,
                                              uint8* compressed_data,
                                              compr_idxs<IndexType>& idxs)
{
    while (row_mat_data != idxs.row) {
        idxs.row++;
        idxs.col = 0;
        set_value_compressed_data_and_increment<uint8>(
            compressed_data, idxs.shf, cst_mark_end_row);
    }
    return (col_mat_data - idxs.col);
}


// Detects if a block is complete, updating idxs
template <typename IndexType>
inline void get_detect_endblock(const IndexType block_size,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == block_size) {
        idxs.nblk = 0;
        idxs.blk++;
    }
}


// Detects if a block is complete, updating idxs and writing in block_offsets
template <typename IndexType>
inline void put_detect_endblock(size_type* block_offsets,
                                const IndexType block_size,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == block_size) {
        idxs.nblk = 0;
        idxs.blk++;
        block_offsets[idxs.blk] = idxs.shf;
    }
}


// Detects if a block is complete, updating idxs
template <typename IndexType>
inline void cnt_detect_endblock(const IndexType block_size,
                                compr_idxs<IndexType>& idxs)
{
    if (idxs.nblk == block_size) {
        idxs.nblk = 0;
        idxs.blk++;
    }
}


/*
 *  Methods for managing group compression objects
 */


// Updates grp_idxs, assuming that (row,col) is added
template <typename IndexType>
inline void proc_group_keys(const IndexType row, const IndexType col,
                            const compr_idxs<IndexType>& idxs,
                            compr_grp_idxs<IndexType>& grp_idxs)
{
    if (idxs.nblk == 0) {
        grp_idxs = {};
        grp_idxs.row_frst = row;
        grp_idxs.col_frst = col;
    }
    if (row != grp_idxs.row_frst) {
        grp_idxs.rows_cols |= type_mask_rows_multiple;
        if (row > (grp_idxs.row_frst + grp_idxs.row_diff)) {
            grp_idxs.row_diff = row - grp_idxs.row_frst;
        }
    }
    if (col < grp_idxs.col_frst) {
        grp_idxs.col_diff += (grp_idxs.col_frst - col);
        grp_idxs.col_frst = col;
    } else if (col > (grp_idxs.col_frst + grp_idxs.col_diff)) {
        grp_idxs.col_diff = col - grp_idxs.col_frst;
    }
}


// Adapts idxs according to values in grp_idxs
template <typename IndexType, typename ValueType>
inline void cnt_group_keys(const IndexType block_size,
                           const compr_grp_idxs<IndexType>& grp_idxs,
                           compr_idxs<IndexType>& idxs)
{
    if (grp_idxs.row_diff > 0) {
        idxs.shf += ((grp_idxs.row_diff > cst_max_size_small_idxs_row)
                         ? sizeof(uint16)
                         : sizeof(uint8)) *
                    block_size;
    }
    if (grp_idxs.col_diff <= cst_max_col_diff_small) {
        idxs.shf += block_size * sizeof(uint8);
    } else if (grp_idxs.col_diff <= cst_max_col_diff_medium) {
        idxs.shf += block_size * sizeof(uint16);
    } else if (grp_idxs.col_diff <= cst_max_col_diff_large) {
        idxs.shf += block_size * sizeof(uint32);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    idxs.shf += sizeof(ValueType) * block_size;
}


// Adapts idxs and grp_idxs, returning val from compressed_data
template <typename IndexType, typename ValueType>
inline void get_group_position_value(const uint8* compressed_data,
                                     compr_grp_idxs<IndexType>& grp_idxs,
                                     compr_idxs<IndexType>& idxs,
                                     ValueType& val)
{
    idxs.row = grp_idxs.row_frst;
    idxs.col = grp_idxs.col_frst;
    if (grp_idxs.is_multi_row()) {
        if (grp_idxs.is_row_16bits()) {
            idxs.row += get_value_compressed_data_and_increment<uint16>(
                compressed_data, grp_idxs.shf_row);
        } else {
            idxs.row += get_value_compressed_data_and_increment<uint8>(
                compressed_data, grp_idxs.shf_row);
        }
    }
    if (grp_idxs.is_column_8bits()) {
        idxs.col += get_value_compressed_data_and_increment<uint8>(
            compressed_data, grp_idxs.shf_col);
    } else if (grp_idxs.is_column_16bits()) {
        idxs.col += get_value_compressed_data_and_increment<uint16>(
            compressed_data, grp_idxs.shf_col);
    } else {
        idxs.col += get_value_compressed_data_and_increment<uint32>(
            compressed_data, grp_idxs.shf_col);
    }
    val = get_value_compressed_data_and_increment<ValueType>(compressed_data,
                                                             grp_idxs.shf_val);
}


// Adapts idxs and grp_idxs, getting val from compressed_data
// Then applies finalize_op on val, writing it on compressed_data and returning
// it
template <typename IndexType, typename ValueType, typename Callable>
inline void get_group_position_value_put(uint8* compressed_data,
                                         compr_grp_idxs<IndexType>& grp_idxs,
                                         compr_idxs<IndexType>& idxs,
                                         ValueType& val, Callable finalize_op)
{
    idxs.row = grp_idxs.row_frst;
    idxs.col = grp_idxs.col_frst;
    if (grp_idxs.is_multi_row()) {
        if (grp_idxs.is_row_16bits()) {
            idxs.row += get_value_compressed_data_and_increment<uint16>(
                compressed_data, grp_idxs.shf_row);
        } else {
            idxs.row += get_value_compressed_data_and_increment<uint8>(
                compressed_data, grp_idxs.shf_row);
        }
    }
    if (grp_idxs.is_column_8bits()) {
        idxs.col += get_value_compressed_data_and_increment<uint8>(
            compressed_data, grp_idxs.shf_col);
    } else if (grp_idxs.is_column_16bits()) {
        idxs.col += get_value_compressed_data_and_increment<uint16>(
            compressed_data, grp_idxs.shf_col);
    } else {
        idxs.col += get_value_compressed_data_and_increment<uint32>(
            compressed_data, grp_idxs.shf_col);
    }
    val =
        get_value_compressed_data<ValueType>(compressed_data, grp_idxs.shf_val);
    val = finalize_op(val);
    set_value_compressed_data_and_increment<ValueType>(compressed_data,
                                                       grp_idxs.shf_val, val);
}


// Writes (rows_grp, cols_grp, vals_grp) on compressed_data, updating idxs and
// grp_idxs and returning type in which the formats are described
template <typename IndexType, typename ValueType>
inline uint8 write_compressed_data_grp_type(
    compr_idxs<IndexType>& idxs, const compr_grp_idxs<IndexType>& grp_idxs,
    const array<IndexType>& rows_grp, const array<IndexType>& cols_grp,
    const array<ValueType>& vals_grp, uint8* compressed_data)
{
    uint8 type_grp = {};

    // Counting bytes to write group on result
    if (grp_idxs.is_multi_row()) {
        if (grp_idxs.row_diff > cst_max_size_small_idxs_row) {
            for (IndexType j = 0; j < idxs.nblk; j++) {
                uint16 row_diff =
                    rows_grp.get_const_data()[j] - grp_idxs.row_frst;
                set_value_compressed_data_and_increment<uint16>(
                    compressed_data, idxs.shf, row_diff);
            }
            type_grp |= type_mask_rows_16bits;
        } else {
            for (IndexType j = 0; j < idxs.nblk; j++) {
                uint8 row_diff =
                    rows_grp.get_const_data()[j] - grp_idxs.row_frst;
                set_value_compressed_data_and_increment<uint8>(
                    compressed_data, idxs.shf, row_diff);
            }
        }
        type_grp |= type_mask_rows_multiple;
    }
    if (grp_idxs.col_diff <= cst_max_col_diff_small) {
        for (IndexType j = 0; j < idxs.nblk; j++) {
            uint8 col_diff = cols_grp.get_const_data()[j] - grp_idxs.col_frst;
            set_value_compressed_data_and_increment<uint8>(compressed_data,
                                                           idxs.shf, col_diff);
        }
        type_grp |= type_mask_cols_8bits;
    } else if (grp_idxs.col_diff <= cst_max_col_diff_medium) {
        for (IndexType j = 0; j < idxs.nblk; j++) {
            uint16 col_diff = cols_grp.get_const_data()[j] - grp_idxs.col_frst;
            set_value_compressed_data_and_increment<uint16>(compressed_data,
                                                            idxs.shf, col_diff);
        }
        type_grp |= type_mask_cols_16bits;
    } else if (grp_idxs.col_diff <= cst_max_col_diff_large) {
        for (IndexType j = 0; j < idxs.nblk; j++) {
            uint32 col_diff = cols_grp.get_const_data()[j] - grp_idxs.col_frst;
            set_value_compressed_data_and_increment<uint32>(compressed_data,
                                                            idxs.shf, col_diff);
        }
    } else {
        GKO_NOT_IMPLEMENTED;
    }
    for (IndexType j = 0; j < idxs.nblk; j++) {
        ValueType val = vals_grp.get_const_data()[j];
        set_value_compressed_data_and_increment<ValueType>(compressed_data,
                                                           idxs.shf, val);
    }

    return type_grp;
}


// Copy [rows_grp, cols_grp, vals_grp] from compressed_data_src to
// 		compressed_data_res, but applying finalize_op() on vals_grp.
// The corresponding idxs and grp_idxs are updated
template <typename IndexType, typename ValueType_src, typename ValueType_res,
          typename Callable>
inline void write_compressed_data_grp(
    compr_idxs<IndexType>& idxs_src,
    const compr_grp_idxs<IndexType>& grp_idxs_src,
    const IndexType block_size_local_src, const uint8* compressed_data_src,
    compr_idxs<IndexType>& idxs_res,
    const compr_grp_idxs<IndexType>& grp_idxs_res,
    const IndexType block_size_local_res, uint8* compressed_data_res,
    Callable finalize_op)
{
    ValueType_src val_src;
    ValueType_res val_res;
    if (grp_idxs_src.is_multi_row()) {
        if (grp_idxs_src.is_row_16bits()) {
            copy_array_compressed_data_and_increment<uint16>(
                compressed_data_res, idxs_res.shf, compressed_data_src,
                idxs_src.shf, block_size_local_src);
        } else {
            copy_array_compressed_data_and_increment<uint8>(
                compressed_data_res, idxs_res.shf, compressed_data_src,
                idxs_src.shf, block_size_local_src);
        }
    }
    if (grp_idxs_src.is_column_8bits()) {
        copy_array_compressed_data_and_increment<uint8>(
            compressed_data_res, idxs_res.shf, compressed_data_src,
            idxs_src.shf, block_size_local_src);
    } else if (grp_idxs_src.is_column_16bits()) {
        copy_array_compressed_data_and_increment<uint16>(
            compressed_data_res, idxs_res.shf, compressed_data_src,
            idxs_src.shf, block_size_local_src);
    } else {
        copy_array_compressed_data_and_increment<uint32>(
            compressed_data_res, idxs_res.shf, compressed_data_src,
            idxs_src.shf, block_size_local_src);
    }
    if (true) {
        for (IndexType i = 0; i < block_size_local_res; i++) {
            val_src = get_value_compressed_data_and_increment<ValueType_src>(
                compressed_data_src, idxs_src.shf);
            val_res = finalize_op(val_src);
            set_value_compressed_data_and_increment<ValueType_res>(
                compressed_data_res, idxs_res.shf, val_res);
        }
    }
}


}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_HELPER_HPP_
