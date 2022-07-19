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

#ifndef GKO_CORE_MATRIX_BCCOO_HELPER_HPP_
#define GKO_CORE_MATRIX_BCCOO_HELPER_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


#include "core/base/unaligned_access.hpp"
#include "core/matrix/bccoo_aux_structs.hpp"

namespace gko {


/*
 *  Routines for mananing bccoo objects
 */


inline void cnt_next_position(const size_type col_src_res, size_type& shf,
                              size_type& col)
{
    if (col_src_res < 0xFD) {
        shf++;
    } else if (col_src_res < 0xFFFF) {
        shf += sizeof(uint16) + 1;
    } else {
        shf += sizeof(uint32) + 1;
    }
    col += col_src_res;
}


template <typename ValueType>
inline void cnt_next_position_value(const size_type col_src_res, size_type& shf,
                                    size_type& col, const ValueType val,
                                    size_type& nblk)
{
    cnt_next_position(col_src_res, shf, col);
    shf += sizeof(ValueType);
    nblk++;
}


inline void get_next_position(const uint8* chunk_data, const uint8 ind,
                              size_type& shf, size_type& col)
{
    if (ind < 0xFD) {
        col += ind;
        shf++;
    } else if (ind == 0xFD) {
        shf++;
        col += get_value_chunk<uint16>(chunk_data, shf);
        shf += sizeof(uint16);
    } else {
        shf++;
        col += get_value_chunk<uint32>(chunk_data, shf);
        shf += sizeof(uint32);
    }
}


template <typename ValueType>
inline void get_next_position_value(const uint8* chunk_data, size_type& nblk,
                                    const uint8 ind, size_type& shf,
                                    size_type& col, ValueType& val)
{
    get_next_position(chunk_data, ind, shf, col);
    val = get_value_chunk<ValueType>(chunk_data, shf);
    shf += sizeof(ValueType);
    nblk++;
}


template <typename ValueType, typename Callable>
inline void get_next_position_value_put(uint8* chunk_data, size_type& nblk,
                                        const uint8 ind, size_type& shf,
                                        size_type& col, ValueType& val,
                                        Callable finalize_op)
{
    get_next_position(chunk_data, ind, shf, col);
    val = get_value_chunk<ValueType>(chunk_data, shf);
    val = finalize_op(val);
    set_value_chunk<ValueType>(chunk_data, shf, val);
    shf += sizeof(ValueType);
    nblk++;
}


inline void put_next_position(uint8* chunk_data, const size_type col_src_res,
                              size_type& shf, size_type& col)
{
    if (col_src_res < 0xFD) {
        set_value_chunk<uint8>(chunk_data, shf, col_src_res);
        col += col_src_res;
        shf++;
    } else if (col_src_res < 0xFFFF) {
        set_value_chunk<uint8>(chunk_data, shf, 0xFD);
        shf++;
        set_value_chunk<uint16>(chunk_data, shf, col_src_res);
        col += col_src_res;
        shf += sizeof(uint16);
    } else {
        set_value_chunk<uint8>(chunk_data, shf, 0xFE);
        shf++;
        set_value_chunk<uint32>(chunk_data, shf, col_src_res);
        col += col_src_res;
        shf += sizeof(uint32);
    }
}


template <typename ValueType>
inline void put_next_position_value(uint8* chunk_data, size_type& nblk,
                                    const size_type col_src_res, size_type& shf,
                                    size_type& col, const ValueType val)
{
    put_next_position(chunk_data, col_src_res, shf, col);
    set_value_chunk<ValueType>(chunk_data, shf, val);
    shf += sizeof(ValueType);
    nblk++;
}


template <typename IndexType>
inline void get_detect_newblock(const IndexType* rows_data,
                                const IndexType* offsets_data, size_type nblk,
                                size_type blk, size_type& shf, size_type& row,
                                size_type& col)
{
    if (nblk == 0) {
        row = rows_data[blk];
        col = 0;
        shf = offsets_data[blk];
    }
}


template <typename IndexType>
inline void put_detect_newblock(IndexType* rows_data, const size_type nblk,
                                const size_type blk, size_type& row,
                                const size_type row_src_res, size_type& col)
{
    if (nblk == 0) {
        row += row_src_res;
        col = 0;
        rows_data[blk] = row;
    }
}


template <typename IndexType>
inline void put_detect_newblock(uint8* chunk_data, IndexType* rows_data,
                                const size_type nblk, const size_type blk,
                                size_type& shf, size_type& row,
                                const size_type row_src_res, size_type& col)
{
    if (nblk == 0) {
        row += row_src_res;
        col = 0;
        rows_data[blk] = row;
    } else if (row_src_res != 0) {  // new row
        row += row_src_res;
        col = 0;
        set_value_chunk<uint8>(chunk_data, shf, 0xFF);
        shf++;
    }
}


inline void cnt_detect_newblock(const size_type nblk, size_type& shf,
                                size_type& row, const size_type row_src_res,
                                size_type& col)
{
    if (nblk == 0) {
        row += row_src_res;
        col = 0;
    } else if (row_src_res != 0) {  // new row
        row += row_src_res;
        col = 0;
        shf += row_src_res;
    }
}


template <typename IndexType>
inline void get_detect_newblock_csr(const IndexType* rows_data,
                                    const IndexType* offsets_data,
                                    size_type nblk, size_type blk,
                                    IndexType* row_ptrs, size_type pos,
                                    size_type& shf, size_type& row,
                                    size_type& col)
{
    if (nblk == 0) {
        if (row != rows_data[blk]) {
            row = rows_data[blk];
            row_ptrs[row] = pos;
        }
        col = 0;
        shf = offsets_data[blk];
    }
}


inline size_type cnt_position_newrow_mat_data(const size_type row_mat_data,
                                              const size_type col_mat_data,
                                              size_type& shf, size_type& row,
                                              size_type& col)
{
    if (row_mat_data != row) {
        shf += row_mat_data - row;
        row = row_mat_data;
        col = 0;
    }
    return (col_mat_data - col);
}


inline uint8 get_position_newrow(const uint8* chunk_data, size_type& shf,
                                 size_type& row, size_type& col)
{
    uint8 ind = get_value_chunk<uint8>(chunk_data, shf);
    while (ind == 0xFF) {
        row++;
        shf++;
        col = 0;
        ind = get_value_chunk<uint8>(chunk_data, shf);
    }
    return ind;
}


template <typename IndexType>
inline uint8 get_position_newrow_csr(const uint8* chunk_data,
                                     IndexType* row_ptrs, size_type pos,
                                     size_type& shf, size_type& row,
                                     size_type& col)
{
    uint8 ind = get_value_chunk<uint8>(chunk_data, shf);
    while (ind == 0xFF) {
        row++;
        col = 0;
        row_ptrs[row] = pos;
        shf++;
        ind = get_value_chunk<uint8>(chunk_data, shf);
    }
    return ind;
}


template <typename IndexType>
inline uint8 get_position_newrow_put(
    const uint8* chunk_data_src, size_type& shf_src, size_type& row_src,
    size_type& col_src, uint8* chunk_data_res, const size_type nblk_res,
    const size_type blk_res, IndexType* rows_data_res, size_type& shf_res,
    size_type& row_res, size_type& col_res)
{
    uint8 ind_src = get_value_chunk<uint8>(chunk_data_src, shf_src);
    while (ind_src == 0xFF) {
        row_src++;
        col_src = 0;
        shf_src++;
        ind_src = get_value_chunk<uint8>(chunk_data_src, shf_src);
        row_res++;
        col_res = 0;
        if (nblk_res == 0) {
            rows_data_res[blk_res] = row_res;
        } else {
            set_value_chunk<uint8>(chunk_data_res, shf_res, 0xFF);
            shf_res++;
        }
    }
    return ind_src;
}


inline size_type put_position_newrow_mat_data(const size_type row_mat_data,
                                              const size_type col_mat_data,
                                              uint8* chunk_data, size_type& shf,
                                              size_type& row, size_type& col)
{
    while (row_mat_data != row) {
        row++;
        col = 0;
        set_value_chunk<uint8>(chunk_data, shf, 0xFF);
        shf++;
    }
    return (col_mat_data - col);
}


inline void get_detect_endblock(const size_type block_size, size_type& nblk,
                                size_type& blk)
{
    if (nblk == block_size) {
        nblk = 0;
        blk++;
    }
}

template <typename IndexType>
inline void put_detect_endblock(IndexType* offsets_data, const size_type shf,
                                const size_type block_size, size_type& nblk,
                                size_type& blk)
{
    if (nblk == block_size) {
        nblk = 0;
        blk++;
        offsets_data[blk] = shf;
    }
}


inline void cnt_detect_endblock(const size_type block_size, size_type& nblk,
                                size_type& blk)
{
    if (nblk == block_size) {
        nblk = 0;
        blk++;
    }
}


/*
 *  Routines for managing block compression objects
 */


template <typename IndexType>
inline void proc_block_indices(const IndexType row, const IndexType col,
                               const compr_idxs idxs, compr_blk_idxs& blk_idxs)
{
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
}


template <typename ValueType>
inline void cnt_block_indices(const size_type block_size,
                              const compr_blk_idxs blk_idxs, compr_idxs& idxs)
{
    if (blk_idxs.mul_row) idxs.shf += block_size;
    if (blk_idxs.col_dif <= 0xFF) {
        idxs.shf += block_size;
    } else if (blk_idxs.col_dif <= 0xFFFF) {
        idxs.shf += block_size * sizeof(uint16);
    } else {
        idxs.shf += block_size * sizeof(uint32);
    }
    idxs.shf += sizeof(ValueType) * block_size;
}


template <typename IndexType, typename ValueType>
inline void get_block_position_value(const uint8* chunk_data,
                                     compr_blk_idxs& blk_idxs, size_type& row,
                                     size_type& col, ValueType& val)
{
    row = blk_idxs.row_frs;
    col = blk_idxs.col_frs;
    if (blk_idxs.mul_row) {
        row += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_row);
        blk_idxs.shf_row++;
    }
    if (blk_idxs.col_8bits) {
        col += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_col);
        blk_idxs.shf_col++;
    } else if (blk_idxs.col_16bits) {
        col += get_value_chunk<uint16>(chunk_data, blk_idxs.shf_col);
        blk_idxs.shf_col += sizeof(uint16);
    } else {
        col += get_value_chunk<uint32>(chunk_data, blk_idxs.shf_col);
        blk_idxs.shf_col += sizeof(uint32);
    }
    val = get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val);
    blk_idxs.shf_val += sizeof(ValueType);
}


template <typename IndexType, typename ValueType, typename Callable>
inline void get_block_position_value_put(uint8* chunk_data, bool mul_row,
                                         bool col_8bits, bool col_16bits,
                                         size_type row_frs, size_type col_frs,
                                         size_type& row, size_type& col,
                                         ValueType& val, size_type& shf_row,
                                         size_type& shf_col, size_type& shf_val,
                                         Callable finalize_op)
{
    row = row_frs;
    col = col_frs;
    if (mul_row) {
        row += get_value_chunk<uint8>(chunk_data, shf_row);
        shf_row++;
    }
    if (col_8bits) {
        col += get_value_chunk<uint8>(chunk_data, shf_col);
        shf_col++;
    } else if (col_16bits) {
        col += get_value_chunk<uint16>(chunk_data, shf_col);
        shf_col += sizeof(uint16);
    } else {
        col += get_value_chunk<uint32>(chunk_data, shf_col);
        shf_col += sizeof(uint32);
    }
    val = get_value_chunk<ValueType>(chunk_data, shf_val);
    val = finalize_op(val);
    set_value_chunk<ValueType>(chunk_data, shf_val, val);
    shf_val += sizeof(ValueType);
}


template <typename IndexType, typename ValueType, typename Callable>
inline void get_block_position_value_put(uint8* chunk_data,
                                         compr_blk_idxs& blk_idxs,
                                         size_type& row, size_type& col,
                                         ValueType& val, Callable finalize_op)
{
    row = blk_idxs.row_frs;
    col = blk_idxs.col_frs;
    if (blk_idxs.mul_row) {
        row += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_row);
        blk_idxs.shf_row++;
    }
    if (blk_idxs.col_8bits) {
        col += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_col);
        blk_idxs.shf_col++;
    } else if (blk_idxs.col_16bits) {
        col += get_value_chunk<uint16>(chunk_data, blk_idxs.shf_col);
        blk_idxs.shf_col += sizeof(uint16);
    } else {
        col += get_value_chunk<uint32>(chunk_data, blk_idxs.shf_col);
        blk_idxs.shf_col += sizeof(uint32);
    }
    val = get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val);
    val = finalize_op(val);
    set_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, val);
    blk_idxs.shf_val += sizeof(ValueType);
}


template <typename IndexType, typename ValueType>
inline uint8 write_chunk_blk_type(compr_idxs& idxs, compr_blk_idxs blk_idxs,
                                  array<IndexType> rows_blk,
                                  array<IndexType> cols_blk,
                                  array<ValueType> vals_blk, uint8* chunk_data)
{
    uint8 type_blk = {};

    // Counting bytes to write block on result
    if (blk_idxs.mul_row) {
        for (size_type j = 0; j < idxs.nblk; j++) {
            uint8 row_dif = rows_blk.get_data()[j] - blk_idxs.row_frs;
            set_value_chunk<uint8>(chunk_data, idxs.shf, row_dif);
            idxs.shf++;
        }
        type_blk |= cst_rows_multiple;
    }
    if (blk_idxs.col_dif <= 0xFF) {
        for (size_type j = 0; j < idxs.nblk; j++) {
            uint8 col_dif = cols_blk.get_data()[j] - blk_idxs.col_frs;
            set_value_chunk<uint8>(chunk_data, idxs.shf, col_dif);
            idxs.shf++;
        }
        type_blk |= cst_cols_8bits;
    } else if (blk_idxs.col_dif <= 0xFFFF) {
        for (size_type j = 0; j < idxs.nblk; j++) {
            uint16 col_dif = cols_blk.get_data()[j] - blk_idxs.col_frs;
            set_value_chunk<uint16>(chunk_data, idxs.shf, col_dif);
            idxs.shf += sizeof(uint16);
        }
        type_blk |= cst_cols_16bits;
    } else {
        for (size_type j = 0; j < idxs.nblk; j++) {
            uint32 col_dif = cols_blk.get_data()[j] - blk_idxs.col_frs;
            set_value_chunk<uint32>(chunk_data, idxs.shf, col_dif);
            idxs.shf += sizeof(uint32);
        }
    }
    for (size_type j = 0; j < idxs.nblk; j++) {
        ValueType val = vals_blk.get_data()[j];
        set_value_chunk<ValueType>(chunk_data, idxs.shf, val);
        idxs.shf += sizeof(ValueType);
    }

    return type_blk;
}


template <typename ValueType_src, typename ValueType_res, typename Callable>
inline void write_chunk_blk(compr_idxs& idxs_src, compr_blk_idxs blk_idxs_src,
                            const size_type block_size_local_src,
                            const uint8* chunk_data_src, compr_idxs& idxs_res,
                            compr_blk_idxs blk_idxs_res,
                            const size_type block_size_local_res,
                            uint8* chunk_data_res, Callable finalize_op)
{
    ValueType_src val_src;
    ValueType_res val_res;
    if (blk_idxs_src.mul_row) {
        const uint8* rows_blk_src =
            reinterpret_cast<const uint8*>(chunk_data_src + idxs_src.shf);
        uint8* rows_blk_res =
            reinterpret_cast<uint8*>(chunk_data_res + idxs_res.shf);
        for (size_type j = 0; j < block_size_local_src; j++) {
            rows_blk_res[j] = rows_blk_src[j];
        }
        idxs_src.shf += block_size_local_src;
        idxs_res.shf += block_size_local_res;
    }
    if (blk_idxs_src.col_8bits) {
        const uint8* cols_blk_src =
            reinterpret_cast<const uint8*>(chunk_data_src + idxs_src.shf);
        uint8* cols_blk_res =
            reinterpret_cast<uint8*>(chunk_data_res + idxs_res.shf);
        for (size_type j = 0; j < block_size_local_src; j++) {
            cols_blk_res[j] = cols_blk_src[j];
        }
        idxs_src.shf += block_size_local_src;
        idxs_res.shf += block_size_local_res;
    } else if (blk_idxs_src.col_16bits) {
        std::memcpy(
            reinterpret_cast<uint16*>(chunk_data_res + idxs_res.shf),
            reinterpret_cast<const uint16*>(chunk_data_src + idxs_src.shf),
            block_size_local_res * sizeof(uint16));
        idxs_src.shf += block_size_local_src * sizeof(uint16);
        idxs_res.shf += block_size_local_res * sizeof(uint16);
    } else {
        std::memcpy(
            reinterpret_cast<uint32*>(chunk_data_res + idxs_res.shf),
            reinterpret_cast<const uint32*>(chunk_data_src + idxs_src.shf),
            block_size_local_res * sizeof(uint32));
        idxs_src.shf += block_size_local_src * sizeof(uint32);
        idxs_res.shf += block_size_local_res * sizeof(uint32);
    }
    if (true) {
        for (size_type i = 0; i < block_size_local_res; i++) {
            val_src =
                get_value_chunk<ValueType_src>(chunk_data_src, idxs_src.shf);
            val_res = finalize_op(val_src);
            set_value_chunk<ValueType_res>(chunk_data_res, idxs_res.shf,
                                           val_res);
            idxs_src.shf += sizeof(ValueType_src);
            idxs_res.shf += sizeof(ValueType_res);
        }
    }
}


}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_HELPER_HPP_
