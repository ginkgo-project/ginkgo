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

#ifndef GKO_CORE_MATRIX_BCCOO_AUX_STRUCTS_HPP_
#define GKO_CORE_MATRIX_BCCOO_AUX_STRUCTS_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


#include "core/base/unaligned_access.hpp"


namespace gko {
namespace matrix {
namespace bccoo {

/**
 *  Constants used to manage bccoo objects
 */
constexpr uint8 type_mask_rows_cols = 15;
constexpr uint8 type_mask_rows_multiple = 8;
constexpr uint8 type_mask_rows_16bits = 4;
constexpr uint8 type_mask_cols_16bits = 2;
constexpr uint8 type_mask_cols_8bits = 1;


/**
 *  Struct to manage bccoo objects
 */
template <typename IndexType>
struct compr_idxs {
    IndexType nblk;  // position in the block
    IndexType blk;   // active block
    IndexType row;   // row index
    IndexType col;   // column index
    size_type shf;   // shift on the chunk

    GKO_ATTRIBUTES compr_idxs() : nblk(0), blk(0), row(0), col(0), shf(0) {}

    GKO_ATTRIBUTES compr_idxs(IndexType blk_init, size_type shf_init,
                              IndexType row_init = 0, IndexType col_init = 0,
                              IndexType nblk_init = 0)
        : nblk(nblk_init),
          blk(blk_init),
          row(row_init),
          col(col_init),
          shf(shf_init)
    {}
};


/**
 *  Specific struct to manage block compression bccoo objects
 */
template <typename IndexType>
struct compr_blk_idxs {
    IndexType row_frst;  // minimum row index in a block
    IndexType col_frst;  // minimum column index in a block
    IndexType row_diff;  // maximum difference between row indices
                         // in a block
    IndexType col_diff;  // maximum difference between column indices
                         // in a block
    size_type shf_row;   // shift in chunk where the rows vector starts
    size_type shf_col;   // shift in chunk where the cols vector starts
    size_type shf_val;   // shift in chunk where the vals vector starts
    uint8 rows_cols;     // combination of several bool conditions: multi_row,
                         // row_16_bits, column_16_bits, column_8_bits

    GKO_ATTRIBUTES compr_blk_idxs()
        : row_frst(0),
          col_frst(0),
          row_diff(0),
          col_diff(0),
          shf_row(0),
          shf_col(0),
          shf_val(0),
          rows_cols(0)
    {}

    GKO_ATTRIBUTES compr_blk_idxs(const IndexType* rows_data,
                                  const IndexType* cols_data,
                                  const size_type block_size,
                                  const compr_idxs<IndexType>& idxs,
                                  const uint8 type_blk)
        : row_frst(rows_data[idxs.blk]),
          col_frst(cols_data[idxs.blk]),
          row_diff(0),
          col_diff(0),
          shf_row(idxs.shf),
          rows_cols(type_blk & type_mask_rows_cols)
    {
        shf_col =
            idxs.shf +
            ((rows_cols & type_mask_rows_multiple)
                 ? ((rows_cols & type_mask_rows_16bits) ? sizeof(uint16) : 1) *
                       block_size
                 : 0);
        shf_val =
            shf_col + block_size * ((rows_cols & type_mask_cols_8bits)
                                        ? 1
                                        : (rows_cols & type_mask_cols_16bits)
                                              ? sizeof(uint16)
                                              : sizeof(uint32));
    }

    GKO_ATTRIBUTES bool is_multi_row() const
    {
        return (rows_cols & type_mask_rows_multiple);
    }

    GKO_ATTRIBUTES bool is_row_16bits() const
    {
        return (rows_cols & type_mask_rows_16bits);
    }

    GKO_ATTRIBUTES bool is_column_16bits() const
    {
        return (rows_cols & type_mask_cols_16bits);
    }

    GKO_ATTRIBUTES bool is_column_8bits() const
    {
        return (rows_cols & type_mask_cols_8bits);
    }
};


}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_AUX_STRUCTS_HPP_
