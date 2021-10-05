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
    std::memcpy(static_cast<std::int8_t*>(ptr) + start, &value, sizeof(T));
}


/**
 * Returns the value in the m-th byte of ptr, which is adjusting to T class.
 *
 * @return the value in the m-th byte of ptr, which is adjusting to T class.
 */
template <typename T>
T get_value_chunk(const uint8* ptr, size_type start)
{
    T val{};
    std::memcpy(&val, ptr + start, sizeof(T));
    return val;
}


inline void update_bccoo_position(const uint8* chunk_data, size_type& shf,
                                  size_type& row, size_type& col)
{
    uint8 ind = (chunk_data[shf]);
    while (ind == 0xFF) {
        row++;
        shf++;
        col = 0;
        ind = chunk_data[shf];
    }
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


template <typename IndexType>
inline void update_bccoo_position(const IndexType* rows_data,
                                  const IndexType* offsets_data,
                                  const uint8* chunk_data, size_type nblk,
                                  size_type blk, size_type& shf, size_type& row,
                                  size_type& col)
{
    if (nblk == 0) {
        row = rows_data[blk];
        col = 0;
        shf = offsets_data[blk];
    }
    update_bccoo_position(chunk_data, shf, row, col);
}


template <typename T>
void update_bccoo_position_val(const uint8* chunk_data, size_type& shf,
                               size_type& row, size_type& col, T& val)
{
    uint8 ind = (chunk_data[shf]);
    while (ind == 0xFF) {
        row++;
        shf++;
        col = 0;
        ind = chunk_data[shf];
    }
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
    val = get_value_chunk<T>(chunk_data, shf);
    shf += sizeof(T);
}


template <typename IndexType, typename T>
void update_bccoo_position_val(const IndexType* rows_data,
                               const IndexType* offsets_data,
                               const uint8* chunk_data, size_type nblk,
                               size_type blk, size_type& shf, size_type& row,
                               size_type& col, T& val)
{
    if (nblk == 0) {
        row = rows_data[blk];
        col = 0;
        shf = offsets_data[blk];
    }
    update_bccoo_position_val(chunk_data, shf, row, col, val);
}


#define UPDATE 1


#if UPDATE > 1

template <typename IndexType, typename T>
void update_bccoo_position_val(const IndexType* rows_data,
                               const IndexType* offsets_data, uint8* chunk_data,
                               size_type nblk, size_type blk, size_type& shf,
                               size_type& row, size_type& col, T& val,
                               std::function<remove_complex<T>(const T)>& func)
{
#if UPDATE == 2
    if (nblk == 0) {
        row = rows_data[blk];
        col = 0;
        shf = offsets_data[blk];
    }
    uint8 ind = (chunk_data[shf]);
    while (ind == 0xFF) {
        row++;
        shf++;
        col = 0;
        ind = chunk_data[shf];
    }
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
    val = get_value_chunk<T>(chunk_data, shf);
#else
    update_bccoo_position_val(rows_data, offsets_data, chunk_data, nblk, blk,
                              shf, row, col, val);
#endif
    val = func(val);
    set_value_chunk<T>(chunk_data, shf, val);
    shf += sizeof(T);
}

#endif


template <typename T>
inline void update_bccoo_position_copy(const uint8* chunk_dataS,
                                       size_type& shfS, size_type& rowS,
                                       size_type& colS, T* rows_dataR,
                                       size_type& nblkR, size_type& blkR,
                                       uint8* chunk_dataR, size_type& shfR,
                                       size_type& rowR, size_type& colR)
{
    uint8 indS = (chunk_dataS[shfS]);
    while (indS == 0xFF) {
        rowS++;
        colS = 0;
        shfS++;
        indS = chunk_dataS[shfS];
        rowR++;
        colR = 0;
        if (nblkR == 0) {
            rows_dataR[blkR] = rowR;
        } else {
            set_value_chunk<uint8>(chunk_dataR, shfR, 0xFF);
            shfR++;
        }
    }

    if (indS < 0xFD) {
        colS += indS;
        shfS++;
        set_value_chunk<uint8>(chunk_dataR, shfR, indS);
        shfR++;
    } else if (indS == 0xFD) {
        shfS++;
        colS += get_value_chunk<uint16>(chunk_dataS, shfS);
        shfS += 2;
        set_value_chunk<uint8>(chunk_dataR, shfR, 0xFD);
        shfR++;
        set_value_chunk<uint16>(chunk_dataR, shfR, colS - colR);
        colR = colS;
        shfR += 2;
    } else {
        shfS++;
        colS += *(uint32*)(chunk_dataS + shfS);
        shfS += 4;
        set_value_chunk<uint8>(chunk_dataR, shfR, 0xFE);
        shfR++;
        set_value_chunk<uint32>(chunk_dataR, shfR, colS - colR);
        colR = colS;
        shfR += 4;
    }
}


}  // namespace gko

#endif  // GKO_CORE_BASE_UNALIGNED_ACCESS_HPP_
