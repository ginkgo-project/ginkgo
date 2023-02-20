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

namespace gko {


// Routines for block compression objects


template <typename IndexType, typename ValueType>
inline GKO_ATTRIBUTES void get_block_position_value(
    const size_type pos, const uint8* chunk_data, compr_blk_idxs& blk_idxs,
    size_type& row, size_type& col, ValueType& val)
{
    row = blk_idxs.row_frs;
    col = blk_idxs.col_frs;
    if (blk_idxs.mul_row) {
        row += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_row + pos);
    }
    if (blk_idxs.col_8bits) {
        col += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_col + pos);
    } else if (blk_idxs.col_16bits) {
        col += get_value_chunk<uint16>(chunk_data,
                                       blk_idxs.shf_col + pos * sizeof(uint16));
    } else {
        col += get_value_chunk<uint32>(chunk_data,
                                       blk_idxs.shf_col + pos * sizeof(uint32));
    }
    val = get_value_chunk<ValueType>(
        chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType));
}


template <typename IndexType, typename ValueType, typename Closure>
inline GKO_ATTRIBUTES void get_block_position_value_put(
    const size_type pos, uint8* chunk_data, compr_blk_idxs& blk_idxs,
    size_type& row, size_type& col, ValueType& val, Closure finalize_op)
{
    row = blk_idxs.row_frs;
    col = blk_idxs.col_frs;
    if (blk_idxs.mul_row) {
        row += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_row + pos);
    }
    if (blk_idxs.col_8bits) {
        col += get_value_chunk<uint8>(chunk_data, blk_idxs.shf_col + pos);
    } else if (blk_idxs.col_16bits) {
        col += get_value_chunk<uint16>(chunk_data,
                                       blk_idxs.shf_col + pos * sizeof(uint16));
    } else {
        col += get_value_chunk<uint32>(chunk_data,
                                       blk_idxs.shf_col + pos * sizeof(uint32));
    }
    val = get_value_chunk<ValueType>(
        chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType));
    auto new_val = finalize_op(val);
    set_value_chunk<ValueType>(
        chunk_data, blk_idxs.shf_val + pos * sizeof(ValueType), new_val);
}


template <typename IndexType, typename ValueType1, typename ValueType2,
          typename Closure>
inline GKO_ATTRIBUTES void get_block_position_value_put(
    const size_type pos, const uint8* chunk_data_src,
    compr_blk_idxs& blk_idxs_src, uint8* chunk_data_res,
    compr_blk_idxs& blk_idxs_res, size_type& row, size_type& col,
    ValueType1& val, Closure finalize_op)
{
    row = blk_idxs_src.row_frs;
    col = blk_idxs_src.col_frs;
    if (blk_idxs_src.mul_row) {
        auto row_dif =
            get_value_chunk<uint8>(chunk_data_src, blk_idxs_src.shf_row + pos);
        set_value_chunk<uint8>(chunk_data_res, blk_idxs_res.shf_row + pos,
                               row_dif);
        row += row_dif;
    }
    if (blk_idxs_src.col_8bits) {
        auto col_dif =
            get_value_chunk<uint8>(chunk_data_src, blk_idxs_src.shf_col + pos);
        set_value_chunk<uint8>(chunk_data_res, blk_idxs_res.shf_col + pos,
                               col_dif);
        col += col_dif;
    } else if (blk_idxs_src.col_16bits) {
        auto col_dif = get_value_chunk<uint16>(
            chunk_data_src, blk_idxs_src.shf_col + pos * sizeof(uint16));
        set_value_chunk<uint16>(chunk_data_res,
                                blk_idxs_res.shf_col + pos * sizeof(uint16),
                                col_dif);
        col += col_dif;
    } else {
        auto col_dif = get_value_chunk<uint32>(
            chunk_data_src, blk_idxs_src.shf_col + pos * sizeof(uint32));
        set_value_chunk<uint32>(chunk_data_res,
                                blk_idxs_res.shf_col + pos * sizeof(uint32),
                                col_dif);
        col += col_dif;
    }
    val = get_value_chunk<ValueType1>(
        chunk_data_src, blk_idxs_src.shf_val + pos * sizeof(ValueType1));
    auto new_val = finalize_op(val);
    set_value_chunk<ValueType2>(chunk_data_res,
                                blk_idxs_res.shf_val + pos * sizeof(ValueType2),
                                new_val);
}


}  // namespace gko
