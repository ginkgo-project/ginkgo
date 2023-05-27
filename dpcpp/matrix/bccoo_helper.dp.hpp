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

#include "dpcpp/components/segment_scan.dp.hpp"


using namespace gko::matrix::bccoo;


namespace gko {
namespace kernels {
/**
 * @brief DPCPP namespace.
 *
 * @ingroup dpcpp
 */
namespace dpcpp {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


// Routines for block compression objects


template <typename IndexTypeCol, typename IndexType,
          int subgroup_size = config::warp_size, typename ValueType,
          typename Closure>
inline GKO_ATTRIBUTES void loop_block_single_row_spmv(
    const uint8* __restrict__ chunk_data, size_type block_size_local,
    const ValueType* __restrict__ b, const size_type b_stride,
    const size_type column_id, ValueType* __restrict__ c,
    const size_type c_stride, compr_idxs<IndexType>& idxs,
    compr_blk_idxs<IndexType>& blk_idxs, const size_type start_in_blk,
    const size_type jump_in_blk, Closure scale, sycl::nd_item<3> item_ct1)
{
    ValueType temp_val = zero<ValueType>();
    bool new_value = false;
    ValueType val;
    const auto tile_block = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));


    for (size_type pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        idxs.col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(
                chunk_data, blk_idxs.shf_col + pos * sizeof(IndexTypeCol));
        val = get_value_chunk<ValueType>(
            chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType));
        temp_val += val * b[idxs.col * b_stride + column_id];
        new_value = true;
    }
    if (tile_block.any(new_value)) {
        bool is_first_in_segment = segment_scan<subgroup_size>(
            tile_block, blk_idxs.row_frst, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(c[blk_idxs.row_frst * c_stride + column_id]),
                       scale(temp_val));
        }
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          int subgroup_size = config::warp_size, typename ValueType,
          typename Closure>
inline GKO_ATTRIBUTES void loop_block_multi_row_spmv(
    const uint8* __restrict__ chunk_data, size_type block_size_local,
    const ValueType* __restrict__ b, const size_type b_stride,
    const size_type column_id, ValueType* __restrict__ c,
    const size_type c_stride, compr_idxs<IndexType>& idxs,
    compr_blk_idxs<IndexType>& blk_idxs, const size_type start_in_blk,
    const size_type jump_in_blk, Closure scale, sycl::nd_item<3> item_ct1)
{
    //    auto next_row = blk_idxs.row_frst;
    //    auto last_row = blk_idxs.row_frst;
    ValueType temp_val = zero<ValueType>();
    ValueType val;
    bool new_value = false;
    const auto tile_block = group::tiled_partition<subgroup_size>(
        group::this_thread_block(item_ct1));

    auto last_row =
        blk_idxs.row_frst +
        get_value_chunk<IndexTypeRow>(
            chunk_data,
            blk_idxs.shf_row + (block_size_local - 1) * sizeof(IndexTypeRow));
    auto next_row =
        blk_idxs.row_frst +
        get_value_chunk<IndexTypeRow>(
            chunk_data, blk_idxs.shf_row + start_in_blk * sizeof(IndexTypeRow));
    for (size_type pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        idxs.row = next_row;
        idxs.col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(
                chunk_data, blk_idxs.shf_col + pos * sizeof(IndexTypeCol));
        val = get_value_chunk<ValueType>(
            chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType));
        temp_val += val * b[idxs.col * b_stride + column_id];
        new_value = true;
        next_row = ((pos + jump_in_blk) >= block_size_local)
                       ? last_row
                       : blk_idxs.row_frst +
                             get_value_chunk<IndexTypeRow>(
                                 chunk_data,
                                 blk_idxs.shf_row + (pos + jump_in_blk) *
                                                        sizeof(IndexTypeRow));
        // segmented scan
        if (tile_block.any(idxs.row != next_row)) {
            bool is_first_in_segment =
                segment_scan<subgroup_size>(tile_block, idxs.row, &temp_val);
            if (is_first_in_segment) {
                atomic_add(&(c[idxs.row * c_stride + column_id]),
                           scale(temp_val));
            }
            temp_val = zero<ValueType>();
            new_value = false;
        }
    }
    // segmented scan
    if (tile_block.any(new_value)) {
        bool is_first_in_segment =
            segment_scan<subgroup_size>(tile_block, idxs.row, &temp_val);
        if (is_first_in_segment) {
            atomic_add(&(c[idxs.row * c_stride + column_id]), scale(temp_val));
        }
        temp_val = zero<ValueType>();
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
inline void loop_block_single_row_extract(const uint8* chunk_data,
                                          compr_blk_idxs<IndexType>& blk_idxs,
                                          IndexType start_in_blk,
                                          IndexType jump_in_blk,
                                          IndexType block_size_local,
                                          ValueType* __restrict__ diag)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row = blk_idxs.row_frst;
        IndexType col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(chunk_data, blk_idxs.shf_col, pos);
        if (row == col) {
            diag[col] =
                get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        }
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
inline void loop_block_multi_row_extract(const uint8* chunk_data,
                                         compr_blk_idxs<IndexType>& blk_idxs,
                                         IndexType start_in_blk,
                                         IndexType jump_in_blk,
                                         IndexType block_size_local,
                                         ValueType* __restrict__ diag)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row =
            blk_idxs.row_frst +
            get_value_chunk<IndexTypeRow>(chunk_data, blk_idxs.shf_row, pos);
        IndexType col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(chunk_data, blk_idxs.shf_col, pos);
        if (row == col) {
            diag[col] =
                get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        }
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType,
          typename Closure>
inline void loop_block_single_row_absolute(uint8* chunk_data,
                                           compr_blk_idxs<IndexType>& blk_idxs,
                                           IndexType start_in_blk,
                                           IndexType jump_in_blk,
                                           IndexType block_size_local,
                                           Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        ValueType val =
            get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        auto new_val = finalize_op(val);
        set_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos, new_val);
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType, typename Closure>
inline void loop_block_multi_row_absolute(uint8* chunk_data,
                                          compr_blk_idxs<IndexType>& blk_idxs,
                                          IndexType start_in_blk,
                                          IndexType jump_in_blk,
                                          IndexType block_size_local,
                                          Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        ValueType val =
            get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        auto new_val = finalize_op(val);
        set_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos, new_val);
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueTypeSrc,
          typename ValueTypeRes, typename Closure>
inline void loop_block_single_row_absolute(
    const uint8* chunk_data_src, compr_blk_idxs<IndexType>& blk_idxs_src,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    uint8* chunk_data_res, compr_blk_idxs<IndexType>& blk_idxs_res,
    Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexTypeCol col_diff = get_value_chunk<IndexTypeCol>(
            chunk_data_src, blk_idxs_src.shf_col, pos);
        set_value_chunk<IndexTypeCol>(chunk_data_res, blk_idxs_res.shf_col, pos,
                                      col_diff);
        ValueTypeSrc val = get_value_chunk<ValueTypeSrc>(
            chunk_data_src, blk_idxs_src.shf_val, pos);
        ValueTypeRes new_val = finalize_op(val);
        set_value_chunk<ValueTypeRes>(chunk_data_res, blk_idxs_res.shf_val, pos,
                                      new_val);
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueTypeSrc, typename ValueTypeRes, typename Closure>
inline void loop_block_multi_row_absolute(
    const uint8* chunk_data_src, compr_blk_idxs<IndexType>& blk_idxs_src,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    uint8* chunk_data_res, compr_blk_idxs<IndexType>& blk_idxs_res,
    Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexTypeRow row_diff = get_value_chunk<IndexTypeRow>(
            chunk_data_src, blk_idxs_src.shf_row, pos);
        set_value_chunk<IndexTypeRow>(chunk_data_res, blk_idxs_res.shf_row, pos,
                                      row_diff);
        IndexTypeCol col_diff = get_value_chunk<IndexTypeCol>(
            chunk_data_src, blk_idxs_src.shf_col, pos);
        set_value_chunk<IndexTypeCol>(chunk_data_res, blk_idxs_res.shf_col, pos,
                                      col_diff);
        ValueTypeSrc val = get_value_chunk<ValueTypeSrc>(
            chunk_data_src, blk_idxs_src.shf_val, pos);
        ValueTypeRes new_val = finalize_op(val);
        set_value_chunk<ValueTypeRes>(chunk_data_res, blk_idxs_res.shf_val, pos,
                                      new_val);
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
inline void loop_block_single_row_fill_in_coo(
    const uint8* chunk_data, const IndexType blk,
    compr_blk_idxs<IndexType>& blk_idxs, IndexType start_in_blk,
    IndexType jump_in_blk, IndexType block_size, IndexType block_size_local,
    IndexType* __restrict__ rows_idxs, IndexType* __restrict__ cols_idxs,
    ValueType* __restrict__ values)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row = blk_idxs.row_frst;
        IndexType col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(chunk_data, blk_idxs.shf_col, pos);
        ValueType val =
            get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        auto index = blk * block_size + pos;
        rows_idxs[index] = row;
        cols_idxs[index] = col;
        values[index] = val;
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
inline void loop_block_multi_row_fill_in_coo(
    const uint8* chunk_data, const IndexType blk,
    compr_blk_idxs<IndexType>& blk_idxs, IndexType start_in_blk,
    IndexType jump_in_blk, IndexType block_size, IndexType block_size_local,
    IndexType* __restrict__ rows_idxs, IndexType* __restrict__ cols_idxs,
    ValueType* __restrict__ values)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row =
            blk_idxs.row_frst +
            get_value_chunk<IndexTypeRow>(chunk_data, blk_idxs.shf_row, pos);
        IndexType col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(chunk_data, blk_idxs.shf_col, pos);
        ValueType val =
            get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        auto index = blk * block_size + pos;
        rows_idxs[index] = row;
        cols_idxs[index] = col;
        values[index] = val;
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
inline void loop_block_single_row_fill_in_dense(
    const uint8* chunk_data, compr_blk_idxs<IndexType>& blk_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    IndexType stride, ValueType* __restrict__ result)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row = blk_idxs.row_frst;
        IndexType col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(chunk_data, blk_idxs.shf_col, pos);
        ValueType val =
            get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        result[row * stride + col] = val;
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
inline void loop_block_multi_row_fill_in_dense(
    const uint8* chunk_data, compr_blk_idxs<IndexType>& blk_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    IndexType stride, ValueType* __restrict__ result)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row =
            blk_idxs.row_frst +
            get_value_chunk<IndexTypeRow>(chunk_data, blk_idxs.shf_row, pos);
        IndexType col =
            blk_idxs.col_frst +
            get_value_chunk<IndexTypeCol>(chunk_data, blk_idxs.shf_col, pos);
        ValueType val =
            get_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val, pos);
        result[row * stride + col] = val;
    }
}


template <typename IndexType, typename ValueType>
inline GKO_ATTRIBUTES void get_block_position_value(
    const IndexType pos, const uint8* chunk_data,
    compr_blk_idxs<IndexType>& blk_idxs, IndexType& row, IndexType& col,
    ValueType& val)
{
    row = blk_idxs.row_frst;
    col = blk_idxs.col_frst;
    if (blk_idxs.is_multi_row()) {
        if (blk_idxs.is_row_16bits()) {
            row += get_value_chunk<uint16>(chunk_data, blk_idxs.shf_row + pos);
        } else {
            row += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_row + pos);
        }
    }
    if (blk_idxs.is_column_8bits()) {
        col += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_col + pos);
    } else if (blk_idxs.is_column_16bits()) {
        col += get_value_chunk<uint16>(chunk_data,
                                       blk_idxs.shf_col + pos * sizeof(uint16));
    } else {
        col += get_value_chunk<uint32>(chunk_data,
                                       blk_idxs.shf_col + pos * sizeof(uint32));
    }
    val = get_value_chunk<ValueType>(
        chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType));
}


template <typename IndexType, typename ValueType>
inline GKO_ATTRIBUTES void get_block_position_value(
    const size_type pos, const uint8* chunk_data,
    compr_blk_idxs<IndexType>& blk_idxs, compr_idxs<IndexType>& idxs,
    ValueType& val)
{
    idxs.row = blk_idxs.row_frst;
    idxs.col = blk_idxs.col_frst;
    if (blk_idxs.is_multi_row()) {
        if (blk_idxs.is_row_16bits()) {
            idxs.row +=
                get_value_chunk<uint16>(chunk_data, blk_idxs.shf_row + pos);
        } else {
            idxs.row +=
                get_value_chunk<uint8>(chunk_data, blk_idxs.shf_row + pos);
        }
    }
    if (blk_idxs.is_column_8bits()) {
        idxs.col += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_col + pos);
    } else if (blk_idxs.is_column_16bits()) {
        idxs.col += get_value_chunk<uint16>(
            chunk_data, blk_idxs.shf_col + pos * sizeof(uint16));
    } else {
        idxs.col += get_value_chunk<uint32>(
            chunk_data, blk_idxs.shf_col + pos * sizeof(uint32));
    }
    val = get_value_chunk<ValueType>(
        chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType));
}


}  // namespace bccoo
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
