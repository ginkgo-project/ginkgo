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


namespace kernel {


// Routines for group compression objects


template <typename IndexTypeCol, typename IndexType,
          int subwarp_size = config::warp_size, typename ValueType,
          typename Closure>
__forceinline__ __device__ void loop_group_single_row_spmv(
    const uint8* __restrict__ compressed_data, IndexType block_size_local,
    const ValueType* __restrict__ b, const IndexType b_stride,
    const IndexType column_id, ValueType* __restrict__ c,
    const IndexType c_stride, compr_idxs<IndexType>& idxs,
    compr_grp_idxs<IndexType>& grp_idxs, const IndexType start_in_blk,
    const IndexType jump_in_blk, Closure scale)
{
    ValueType temp_val = zero<ValueType>();
    bool new_value = false;
    ValueType val;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());


    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        idxs.col =
            grp_idxs.col_frst +
            get_value_compressed_data<IndexTypeCol>(
                compressed_data, grp_idxs.shf_col + pos * sizeof(IndexTypeCol));
        val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val + pos * sizeof(ValueType));
        temp_val += val * b[idxs.col * b_stride + column_id];
        new_value = true;
    }
    if (tile_block.any(new_value)) {
        bool is_first_in_segment = segment_scan<subwarp_size>(
            tile_block, grp_idxs.row_frst, temp_val,
            [](ValueType a, ValueType b) { return a + b; });
        if (is_first_in_segment) {
            atomic_add(&(c[grp_idxs.row_frst * c_stride + column_id]),
                       scale(temp_val));
        }
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          int subwarp_size = config::warp_size, typename ValueType,
          typename Closure>
__forceinline__ __device__ void loop_group_multi_row_spmv(
    const uint8* __restrict__ compressed_data, IndexType block_size_local,
    const ValueType* __restrict__ b, const IndexType b_stride,
    const IndexType column_id, ValueType* __restrict__ c,
    const IndexType c_stride, compr_idxs<IndexType>& idxs,
    compr_grp_idxs<IndexType>& grp_idxs, const IndexType start_in_blk,
    const IndexType jump_in_blk, Closure scale)
{
    ValueType temp_val = zero<ValueType>();
    ValueType val;
    bool new_value = false;
    const auto tile_block =
        group::tiled_partition<subwarp_size>(group::this_thread_block());

    auto last_row =
        grp_idxs.row_frst +
        get_value_compressed_data<IndexTypeRow>(
            compressed_data,
            grp_idxs.shf_row + (block_size_local - 1) * sizeof(IndexTypeRow));
    auto next_row = grp_idxs.row_frst +
                    get_value_compressed_data<IndexTypeRow>(
                        compressed_data,
                        grp_idxs.shf_row + start_in_blk * sizeof(IndexTypeRow));
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        idxs.row = next_row;
        idxs.col =
            grp_idxs.col_frst +
            get_value_compressed_data<IndexTypeCol>(
                compressed_data, grp_idxs.shf_col + pos * sizeof(IndexTypeCol));
        val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val + pos * sizeof(ValueType));
        temp_val += val * b[idxs.col * b_stride + column_id];
        new_value = true;
        next_row = ((pos + jump_in_blk) >= block_size_local)
                       ? last_row
                       : (grp_idxs.row_frst +
                          get_value_compressed_data<IndexTypeRow>(
                              compressed_data,
                              grp_idxs.shf_row +
                                  (pos + jump_in_blk) * sizeof(IndexTypeRow)));
        // segmented scan
        if (tile_block.any(idxs.row != next_row)) {
            bool is_first_in_segment = segment_scan<subwarp_size>(
                tile_block, idxs.row, temp_val,
                [](ValueType a, ValueType b) { return a + b; });
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
        bool is_first_in_segment = segment_scan<subwarp_size>(
            tile_block, idxs.row, temp_val,
            [](ValueType a, ValueType b) { return a + b; });
        if (is_first_in_segment) {
            atomic_add(&(c[idxs.row * c_stride + column_id]), scale(temp_val));
        }
        temp_val = zero<ValueType>();
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
__forceinline__ __device__ void loop_group_single_row_extract(
    const uint8* compressed_data, compr_grp_idxs<IndexType>& grp_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    ValueType* __restrict__ diag)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row = grp_idxs.row_frst;
        IndexType col =
            grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                    compressed_data, grp_idxs.shf_col, pos);
        if (row == col) {
            diag[col] = get_value_compressed_data<ValueType>(
                compressed_data, grp_idxs.shf_val, pos);
        }
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
__forceinline__ __device__ void loop_group_multi_row_extract(
    const uint8* compressed_data, compr_grp_idxs<IndexType>& grp_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    ValueType* __restrict__ diag)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row =
            grp_idxs.row_frst + get_value_compressed_data<IndexTypeRow>(
                                    compressed_data, grp_idxs.shf_row, pos);
        IndexType col =
            grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                    compressed_data, grp_idxs.shf_col, pos);
        if (row == col) {
            diag[col] = get_value_compressed_data<ValueType>(
                compressed_data, grp_idxs.shf_val, pos);
        }
    }
}


template <typename IndexType, typename ValueType, typename Closure>
__forceinline__ __device__ void loop_group_absolute(
    uint8* compressed_data, compr_grp_idxs<IndexType>& grp_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        ValueType val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val, pos);
        auto new_val = finalize_op(val);
        set_value_compressed_data<ValueType>(compressed_data, grp_idxs.shf_val,
                                             pos, new_val);
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueTypeSrc,
          typename ValueTypeRes, typename Closure>
__forceinline__ __device__ void loop_group_single_row_absolute(
    const uint8* compressed_data_src, compr_grp_idxs<IndexType>& grp_idxs_src,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    uint8* compressed_data_res, compr_grp_idxs<IndexType>& grp_idxs_res,
    Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexTypeCol col_diff = get_value_compressed_data<IndexTypeCol>(
            compressed_data_src, grp_idxs_src.shf_col, pos);
        set_value_compressed_data<IndexTypeCol>(
            compressed_data_res, grp_idxs_res.shf_col, pos, col_diff);
        ValueTypeSrc val = get_value_compressed_data<ValueTypeSrc>(
            compressed_data_src, grp_idxs_src.shf_val, pos);
        ValueTypeRes new_val = finalize_op(val);
        set_value_compressed_data<ValueTypeRes>(
            compressed_data_res, grp_idxs_res.shf_val, pos, new_val);
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueTypeSrc, typename ValueTypeRes, typename Closure>
__forceinline__ __device__ void loop_group_multi_row_absolute(
    const uint8* compressed_data_src, compr_grp_idxs<IndexType>& grp_idxs_src,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    uint8* compressed_data_res, compr_grp_idxs<IndexType>& grp_idxs_res,
    Closure finalize_op)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexTypeRow row_diff = get_value_compressed_data<IndexTypeRow>(
            compressed_data_src, grp_idxs_src.shf_row, pos);
        set_value_compressed_data<IndexTypeRow>(
            compressed_data_res, grp_idxs_res.shf_row, pos, row_diff);
        IndexTypeCol col_diff = get_value_compressed_data<IndexTypeCol>(
            compressed_data_src, grp_idxs_src.shf_col, pos);
        set_value_compressed_data<IndexTypeCol>(
            compressed_data_res, grp_idxs_res.shf_col, pos, col_diff);
        ValueTypeSrc val = get_value_compressed_data<ValueTypeSrc>(
            compressed_data_src, grp_idxs_src.shf_val, pos);
        ValueTypeRes new_val = finalize_op(val);
        set_value_compressed_data<ValueTypeRes>(
            compressed_data_res, grp_idxs_res.shf_val, pos, new_val);
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
__forceinline__ __device__ void loop_group_single_row_fill_in_coo(
    const uint8* compressed_data, const IndexType blk,
    compr_grp_idxs<IndexType>& grp_idxs, IndexType start_in_blk,
    IndexType jump_in_blk, IndexType block_size, IndexType block_size_local,
    IndexType* __restrict__ rows_idxs, IndexType* __restrict__ cols_idxs,
    ValueType* __restrict__ values)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row = grp_idxs.row_frst;
        IndexType col =
            grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                    compressed_data, grp_idxs.shf_col, pos);
        ValueType val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val, pos);
        auto index = blk * block_size + pos;
        rows_idxs[index] = row;
        cols_idxs[index] = col;
        values[index] = val;
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
__forceinline__ __device__ void loop_group_multi_row_fill_in_coo(
    const uint8* compressed_data, const IndexType blk,
    compr_grp_idxs<IndexType>& grp_idxs, IndexType start_in_blk,
    IndexType jump_in_blk, IndexType block_size, IndexType block_size_local,
    IndexType* __restrict__ rows_idxs, IndexType* __restrict__ cols_idxs,
    ValueType* __restrict__ values)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row =
            grp_idxs.row_frst + get_value_compressed_data<IndexTypeRow>(
                                    compressed_data, grp_idxs.shf_row, pos);
        IndexType col =
            grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                    compressed_data, grp_idxs.shf_col, pos);
        ValueType val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val, pos);
        auto index = blk * block_size + pos;
        rows_idxs[index] = row;
        cols_idxs[index] = col;
        values[index] = val;
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
__forceinline__ __device__ void loop_group_single_row_fill_in_dense(
    const uint8* compressed_data, compr_grp_idxs<IndexType>& grp_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    IndexType stride, ValueType* __restrict__ result)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row = grp_idxs.row_frst;
        IndexType col =
            grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                    compressed_data, grp_idxs.shf_col, pos);
        ValueType val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val, pos);
        result[row * stride + col] = val;
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
__forceinline__ __device__ void loop_group_multi_row_fill_in_dense(
    const uint8* compressed_data, compr_grp_idxs<IndexType>& grp_idxs,
    IndexType start_in_blk, IndexType jump_in_blk, IndexType block_size_local,
    IndexType stride, ValueType* __restrict__ result)
{
    for (IndexType pos = start_in_blk; pos < block_size_local;
         pos += jump_in_blk) {
        IndexType row =
            grp_idxs.row_frst + get_value_compressed_data<IndexTypeRow>(
                                    compressed_data, grp_idxs.shf_row, pos);
        IndexType col =
            grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                    compressed_data, grp_idxs.shf_col, pos);
        ValueType val = get_value_compressed_data<ValueType>(
            compressed_data, grp_idxs.shf_val, pos);
        result[row * stride + col] = val;
    }
}


template <typename IndexType, typename ValueType>
__forceinline__ __device__ void get_group_position_value(
    const IndexType pos, const uint8* compressed_data,
    compr_grp_idxs<IndexType>& grp_idxs, IndexType& row, IndexType& col,
    ValueType& val)
{
    row = grp_idxs.row_frst;
    col = grp_idxs.col_frst;
    if (grp_idxs.is_multi_row()) {
        if (grp_idxs.is_row_16bits()) {
            row += get_value_compressed_data<uint16>(
                compressed_data, grp_idxs.shf_row + pos * sizeof(uint16));
        } else {
            row += get_value_compressed_data<uint8>(compressed_data,
                                                    grp_idxs.shf_row + pos);
        }
    }
    if (grp_idxs.is_column_8bits()) {
        col += get_value_compressed_data<uint8>(compressed_data,
                                                grp_idxs.shf_col + pos);
    } else if (grp_idxs.is_column_16bits()) {
        col += get_value_compressed_data<uint16>(
            compressed_data, grp_idxs.shf_col + pos * sizeof(uint16));
    } else {
        col += get_value_compressed_data<uint32>(
            compressed_data, grp_idxs.shf_col + pos * sizeof(uint32));
    }
    val = get_value_compressed_data<ValueType>(
        compressed_data, grp_idxs.shf_val + pos * sizeof(ValueType));
}


template <typename IndexType, typename ValueType>
__forceinline__ __device__ void get_group_position_value(
    const IndexType pos, const uint8* compressed_data,
    compr_grp_idxs<IndexType>& grp_idxs, compr_idxs<IndexType>& idxs,
    ValueType& val)
{
    idxs.row = grp_idxs.row_frst;
    idxs.col = grp_idxs.col_frst;
    if (grp_idxs.is_multi_row()) {
        if (grp_idxs.is_row_16bits()) {
            idxs.row += get_value_compressed_data<uint16>(
                compressed_data, grp_idxs.shf_row + pos * sizeof(uint16));
        } else {
            idxs.row += get_value_compressed_data<uint8>(
                compressed_data, grp_idxs.shf_row + pos);
        }
    }
    if (grp_idxs.is_column_8bits()) {
        idxs.col += get_value_compressed_data<uint8>(compressed_data,
                                                     grp_idxs.shf_col + pos);
    } else if (grp_idxs.is_column_16bits()) {
        idxs.col += get_value_compressed_data<uint16>(
            compressed_data, grp_idxs.shf_col + pos * sizeof(uint16));
    } else {
        idxs.col += get_value_compressed_data<uint32>(
            compressed_data, grp_idxs.shf_col + pos * sizeof(uint32));
    }
    val = get_value_compressed_data<ValueType>(
        compressed_data, grp_idxs.shf_val + pos * sizeof(ValueType));
}


}  // namespace kernel
