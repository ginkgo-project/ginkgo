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

#ifndef GKO_OMP_MATRIX_BCCOO_HELPER_HPP_
#define GKO_OMP_MATRIX_BCCOO_HELPER_HPP_


#include <cstring>
#include <functional>


#include <ginkgo/core/base/types.hpp>


#include "core/base/unaligned_access.hpp"
#include "core/matrix/bccoo_aux_structs.hpp"


using namespace gko::matrix::bccoo;


namespace gko {
namespace kernels {
namespace omp {
namespace bccoo {


/*
 *  Routines for managing bccoo objects
 */


/*
 *  Routines for managing group compression objects
 */


template <typename IndexTypeCol, typename IndexType, typename ValueType>
inline void loop_group_single_row(const uint8* compressed_data,
                                  IndexType block_size_local,
                                  const matrix::Dense<ValueType>* b,
                                  matrix::Dense<ValueType>* c,
                                  compr_idxs<IndexType>& idxs,
                                  compr_grp_idxs<IndexType>& grp_idxs,
                                  ValueType* sumV)
{
    auto num_cols = b->get_size()[1];
    auto row = grp_idxs.row_frst;
    bool new_elm = false;
    ValueType val;

    for (IndexType i = 0; i < block_size_local; i++) {
        idxs.col = grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                           compressed_data, grp_idxs.shf_col);
        grp_idxs.shf_col += sizeof(IndexTypeCol);
        val = get_value_compressed_data<ValueType>(compressed_data,
                                                   grp_idxs.shf_val);
        grp_idxs.shf_val += sizeof(ValueType);
        for (IndexType j = 0; j < num_cols; j++) {
            sumV[j] += val * b->at(idxs.col, j);
        }
        new_elm = true;
    }
    if (new_elm) {
        for (IndexType j = 0; j < num_cols; j++) {
            atomic_add(c->at(row, j), sumV[j]);
            sumV[j] = zero<ValueType>();
        }
    }
}


template <typename IndexTypeCol, typename IndexType, typename ValueType>
inline void loop_group_single_row(
    const uint8* compressed_data, IndexType block_size_local,
    const ValueType alpha_val, const matrix::Dense<ValueType>* b,
    matrix::Dense<ValueType>* c, compr_idxs<IndexType>& idxs,
    compr_grp_idxs<IndexType>& grp_idxs, ValueType* sumV)
{
    auto num_cols = b->get_size()[1];
    auto row = grp_idxs.row_frst;
    bool new_elm = false;
    ValueType val;

    for (IndexType i = 0; i < block_size_local; i++) {
        idxs.col = grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                           compressed_data, grp_idxs.shf_col);
        grp_idxs.shf_col += sizeof(IndexTypeCol);
        val = get_value_compressed_data<ValueType>(compressed_data,
                                                   grp_idxs.shf_val);
        grp_idxs.shf_val += sizeof(ValueType);
        for (IndexType j = 0; j < num_cols; j++) {
            sumV[j] += val * b->at(idxs.col, j);
        }
        new_elm = true;
    }
    if (new_elm) {
        for (IndexType j = 0; j < num_cols; j++) {
            atomic_add(c->at(row, j), alpha_val * sumV[j]);
            sumV[j] = zero<ValueType>();
        }
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
inline void loop_group_multi_row(const uint8* compressed_data,
                                 IndexType block_size_local,
                                 const matrix::Dense<ValueType>* b,
                                 matrix::Dense<ValueType>* c,
                                 compr_idxs<IndexType>& idxs,
                                 compr_grp_idxs<IndexType>& grp_idxs,
                                 ValueType* sumV)
{
    auto num_cols = b->get_size()[1];
    auto row_old = grp_idxs.row_frst;
    bool new_elm = false;
    ValueType val;

    for (IndexType i = 0; i < block_size_local; i++) {
        idxs.row = grp_idxs.row_frst + get_value_compressed_data<IndexTypeRow>(
                                           compressed_data, grp_idxs.shf_row);
        grp_idxs.shf_row += sizeof(IndexTypeRow);
        idxs.col = grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                           compressed_data, grp_idxs.shf_col);
        grp_idxs.shf_col += sizeof(IndexTypeCol);
        val = get_value_compressed_data<ValueType>(compressed_data,
                                                   grp_idxs.shf_val);
        grp_idxs.shf_val += sizeof(ValueType);
        if (row_old != idxs.row) {
            // When a new row ia achieved, the computed values
            // have to be accumulated to c
            for (IndexType j = 0; j < num_cols; j++) {
                atomic_add(c->at(row_old, j), sumV[j]);
                sumV[j] = zero<ValueType>();
            }
            new_elm = false;
        }
        for (IndexType j = 0; j < num_cols; j++) {
            sumV[j] += val * b->at(idxs.col, j);
        }
        new_elm = true;
        row_old = idxs.row;
    }
    if (new_elm) {
        // If some values are processed and not accumulated,
        // the computed values have to be accumulated to c
        for (IndexType j = 0; j < num_cols; j++) {
            atomic_add(c->at(row_old, j), sumV[j]);
            sumV[j] = zero<ValueType>();
        }
    }
}


template <typename IndexTypeRow, typename IndexTypeCol, typename IndexType,
          typename ValueType>
inline void loop_group_multi_row(
    const uint8* compressed_data, IndexType block_size_local,
    const ValueType alpha_val, const matrix::Dense<ValueType>* b,
    matrix::Dense<ValueType>* c, compr_idxs<IndexType>& idxs,
    compr_grp_idxs<IndexType>& grp_idxs, ValueType* sumV)
{
    auto num_cols = b->get_size()[1];
    auto row_old = grp_idxs.row_frst;
    bool new_elm = false;
    ValueType val;

    for (IndexType i = 0; i < block_size_local; i++) {
        idxs.row = grp_idxs.row_frst + get_value_compressed_data<IndexTypeRow>(
                                           compressed_data, grp_idxs.shf_row);
        grp_idxs.shf_row += sizeof(IndexTypeRow);
        idxs.col = grp_idxs.col_frst + get_value_compressed_data<IndexTypeCol>(
                                           compressed_data, grp_idxs.shf_col);
        grp_idxs.shf_col += sizeof(IndexTypeCol);
        val = get_value_compressed_data<ValueType>(compressed_data,
                                                   grp_idxs.shf_val);
        grp_idxs.shf_val += sizeof(ValueType);
        if (row_old != idxs.row) {
            // When a new row ia achieved, the computed values
            // have to be accumulated to c
            for (IndexType j = 0; j < num_cols; j++) {
                atomic_add(c->at(row_old, j), alpha_val * sumV[j]);
                sumV[j] = zero<ValueType>();
            }
            new_elm = false;
        }
        for (IndexType j = 0; j < num_cols; j++) {
            sumV[j] += val * b->at(idxs.col, j);
        }
        new_elm = true;
        row_old = idxs.row;
    }
    if (new_elm) {
        // If some values are processed and not accumulated,
        // the computed values have to be accumulated to c
        for (IndexType j = 0; j < num_cols; j++) {
            atomic_add(c->at(row_old, j), alpha_val * sumV[j]);
            sumV[j] = zero<ValueType>();
        }
    }
}


}  // namespace bccoo
}  // namespace omp
}  // namespace kernels
}  // namespace gko


#endif  // GKO_OMP_MATRIX_BCCOO_HELPER_HPP_
