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
constexpr uint8 cst_rows_multiple = 1;
constexpr uint8 cst_rows_16bits = 2;
constexpr uint8 cst_cols_8bits = 4;
constexpr uint8 cst_cols_16bits = 8;


/**
 *  Struct to manage bccoo objects
 */
typedef struct compr_idxs {
    size_type nblk;  // position in the block
    size_type blk;   // active block
    size_type row;   // row index
    size_type col;   // column index
    size_type shf;   // shift on the chunk
} compr_idxs;


/**
 *  Specific struct to manage block compression bccoo objects
 */
typedef struct compr_blk_idxs {
    size_type row_frs;  // minimum row index in a block
    size_type col_frs;  // minimum column index in a block
    size_type row_dif;  // maximum difference between row indices
                        // in a block
    size_type col_dif;  // maximum difference between column indices
                        // in a block
    size_type shf_row;  // shift in chunk where the rows vector starts
    size_type shf_col;  // shift in chunk where the cols vector starts
    size_type shf_val;  // shift in chunk where the vals vector starts
    bool mul_row;       // determines if the block includes elements
                        // of several rows
    bool row_16bits;    // determines that row_dif is greater than 0xFF
    bool col_8bits;     // determines that col_dif is lower than 0x100
    bool col_16bits;    // determines that col_dif is lower than 0x10000
} compr_blk_idxs;


/*
 *  Routines for managing bccoo objects
 */


/*
 *  Routines for managing block compression objects
 */


/**
 *  This routine initializes a compr_blk_idxs object from the information
 *  included in idxs and type_blk, making easier the management of
 *   a block compression bccoo object
 */
template <typename IndexType>
inline GKO_ATTRIBUTES void init_block_indices(const IndexType* rows_data,
                                              const IndexType* cols_data,
                                              const size_type block_size,
                                              const compr_idxs idxs,
                                              const uint8 type_blk,
                                              compr_blk_idxs& blk_idxs)
{
    blk_idxs.mul_row = type_blk & cst_rows_multiple;
    blk_idxs.row_16bits = type_blk & cst_rows_16bits;
    blk_idxs.col_8bits = type_blk & cst_cols_8bits;
    blk_idxs.col_16bits = type_blk & cst_cols_16bits;

    blk_idxs.row_frs = rows_data[idxs.blk];
    blk_idxs.col_frs = cols_data[idxs.blk];
    blk_idxs.shf_row = blk_idxs.shf_col = idxs.shf;
    if (blk_idxs.mul_row)
        blk_idxs.shf_col +=
            ((blk_idxs.row_16bits) ? sizeof(uint16) : 1) * block_size;
    if (blk_idxs.col_8bits) {
        blk_idxs.shf_val = blk_idxs.shf_col + block_size;
    } else if (blk_idxs.col_16bits) {
        blk_idxs.shf_val = blk_idxs.shf_col + block_size * sizeof(uint16);
    } else {
        blk_idxs.shf_val = blk_idxs.shf_col + block_size * sizeof(uint32);
    }
}

}  // namespace bccoo
}  // namespace matrix
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BCCOO_AUX_STRUCTS_HPP_
