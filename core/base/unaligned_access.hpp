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

#ifndef GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_
#define GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


namespace gko {


/**
 * Copies the value in the m-th byte of ptr.
 *
 * @copy the value in the m-th byte of ptr.
 */
template <typename T>
void set_value_chunk(void* ptr, size_type start, T value)
{
    //    std::memcpy(static_cast<std::int8_t *>(ptr) + start, &value,
    //    sizeof(T)); std::memcpy(ptr + start, &value, sizeof(T));
    std::memcpy(static_cast<unsigned char*>(ptr) + start, &value, sizeof(T));
}


/**
 * Returns the value in the m-th byte of ptr, which is adjusting to T class.
 *
 * @return the value in the m-th byte of ptr, which is adjusting to T class.
 */
template <typename T>
// T get_value_chunk(const uint8 *ptr, size_type start)
T get_value_chunk(const void* ptr, size_type start)
{
    T val{};
    //    std::memcpy(&val, ptr + start, sizeof(T));
    //    std::memcpy(&val, static_cast<std::int8_t *>(ptr) + start, sizeof(T));
    std::memcpy(&val, static_cast<const unsigned char*>(ptr) + start,
                sizeof(T));
    return val;
}


// inline void cnt_next_position(const uint8 ind, size_type &shf)
inline void cnt_next_position(const size_type colRS, size_type& shf,
                              size_type& col)
{
    if (colRS < 0xFD) {
        shf++;
    } else if (colRS < 0xFFFF) {
        shf += 3;
    } else {
        shf += 5;
    }
    col += colRS;
}


template <typename ValueType>
inline void cnt_next_position_value(
    const size_type colRS, size_type& shf,
    // inline void cnt_next_position_value(const int8 ind, size_type &shf,
    size_type& col, const ValueType val, size_type& nblk)
{
    cnt_next_position(colRS, shf, col);
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
        shf += 2;
    } else {
        shf++;
        col += get_value_chunk<uint32>(chunk_data, shf);
        shf += 4;
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


inline void put_next_position(uint8* chunk_data, const size_type colRS,
                              size_type& shf, size_type& col)
{
    if (colRS < 0xFD) {
        set_value_chunk<uint8>(chunk_data, shf, colRS);
        col += colRS;
        shf++;
    } else if (colRS < 0xFFFF) {
        set_value_chunk<uint8>(chunk_data, shf, 0xFD);
        shf++;
        set_value_chunk<uint16>(chunk_data, shf, colRS);
        col += colRS;
        shf += 2;
    } else {
        set_value_chunk<uint8>(chunk_data, shf, 0xFE);
        shf++;
        set_value_chunk<uint32>(chunk_data, shf, colRS);
        col += colRS;
        shf += 4;
    }
}


template <typename ValueType>
inline void put_next_position_value(uint8* chunk_data, size_type& nblk,
                                    const size_type colRS, size_type& shf,
                                    size_type& col, const ValueType val)
{
    put_next_position(chunk_data, colRS, shf, col);
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
                                const size_type rowRS, size_type& col)
{
    if (nblk == 0) {
        row += rowRS;
        col = 0;
        rows_data[blk] = row;
    }
}


template <typename IndexType>
inline void put_detect_newblock(uint8* chunk_data, IndexType* rows_data,
                                const size_type nblk, const size_type blk,
                                size_type& shf, size_type& row,
                                const size_type rowRS, size_type& col)
{
    if (nblk == 0) {
        row += rowRS;
        col = 0;
        rows_data[blk] = row;
    } else if (rowRS != 0) {  // new row
        row += rowRS;
        col = 0;
        set_value_chunk<uint8>(chunk_data, shf, 0xFF);
        shf++;
    }
}


inline void cnt_detect_newblock(const size_type nblk, size_type& shf,
                                size_type& row, const size_type rowRS,
                                size_type& col)
{
    if (nblk == 0) {
        row += rowRS;
        col = 0;
    } else if (rowRS != 0) {  // new row
        row += rowRS;
        col = 0;
        shf += rowRS;
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


inline size_type cnt_position_newrow_mat_data(const size_type rowMD,
                                              const size_type colMD,
                                              size_type& shf, size_type& row,
                                              size_type& col)
{
    if (rowMD != row) {
        shf += rowMD - row;
        row = rowMD;
        col = 0;
    }
    return (colMD - col);
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
inline uint8 get_position_newrow_put(const uint8* chunk_dataS, size_type& shfS,
                                     size_type& rowS, size_type& colS,
                                     uint8* chunk_dataR, const size_type nblkR,
                                     const size_type blkR,
                                     IndexType* rows_dataR, size_type& shfR,
                                     size_type& rowR, size_type& colR)
{
    uint8 indS = get_value_chunk<uint8>(chunk_dataS, shfS);
    while (indS == 0xFF) {
        rowS++;
        colS = 0;
        shfS++;
        indS = get_value_chunk<uint8>(chunk_dataS, shfS);
        rowR++;
        colR = 0;
        if (nblkR == 0) {
            rows_dataR[blkR] = rowR;
        } else {
            set_value_chunk<uint8>(chunk_dataR, shfR, 0xFF);
            shfR++;
        }
    }
    return indS;
}


inline size_type put_position_newrow_mat_data(const size_type rowMD,
                                              const size_type colMD,
                                              uint8* chunk_data, size_type& shf,
                                              size_type& row, size_type& col)
{
    while (rowMD != row) {
        row++;
        col = 0;
        set_value_chunk<uint8>(chunk_data, shf, 0xFF);
        shf++;
    }
    return (colMD - col);
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


}  // namespace gko


#endif  // GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_
