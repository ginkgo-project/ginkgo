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

#include "core/matrix/bccoo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The Reference namespace.
 *
 * @ingroup reference
 */
namespace reference {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


const int GKO_BCCOO_ROWS_MULTIPLE = 1;
const int GKO_BCCOO_COLS_8BITS = 2;
const int GKO_BCCOO_COLS_16BITS = 4;


void get_default_block_size(std::shared_ptr<const ReferenceExecutor> exec,
                            size_type* block_size)
{
    *block_size = 10;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    for (size_type i = 0; i < c->get_num_stored_elements(); i++) {
        c->at(i) = zero<ValueType>();
    }
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    auto beta_val = beta->at(0, 0);
    for (size_type i = 0; i < c->get_num_stored_elements(); i++) {
        c->at(i) *= beta_val;
    }
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto* rows_data = a->get_const_rows();
    auto* offsets_data = a->get_const_offsets();
    auto* chunk_data = a->get_const_chunk();
    auto num_stored_elements = a->get_num_stored_elements();
    auto block_size = a->get_block_size();
    auto num_cols = b->get_size()[1];

    // Computation of chunk
    size_type nblk = 0;
    size_type blk = 0;
    size_type col = 0;
    size_type row = 0;
    size_type shf = 0;
    ValueType val;
    for (size_type i = 0; i < num_stored_elements; i++) {
        get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row, col);
        uint8 ind = get_position_newrow(chunk_data, shf, row, col);
        get_next_position_value(chunk_data, nblk, ind, shf, col, val);
        get_detect_endblock(block_size, nblk, blk);
        for (size_type j = 0; j < num_cols; j++) {
            c->at(row, j) += val * b->at(col, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    auto* rows_data = a->get_const_rows();
    auto* offsets_data = a->get_const_offsets();
    auto* chunk_data = a->get_const_chunk();
    auto num_stored_elements = a->get_num_stored_elements();
    auto block_size = a->get_block_size();
    auto alpha_val = alpha->at(0, 0);
    auto num_cols = b->get_size()[1];

    // Computation of chunk
    size_type nblk = 0;
    size_type blk = 0;
    size_type col = 0;
    size_type row = 0;
    size_type shf = 0;
    ValueType val;
    for (size_type i = 0; i < num_stored_elements; i++) {
        get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row, col);
        uint8 ind = get_position_newrow(chunk_data, shf, row, col);
        get_next_position_value(chunk_data, nblk, ind, shf, col, val);
        get_detect_endblock(block_size, nblk, blk);
        for (size_type j = 0; j < num_cols; j++) {
            c->at(row, j) += alpha_val * val * b->at(col, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::bccoo
                    : compression commpress_res,
                      size_type* mem_size) GKO_NOT_IMPLEMENTED;
/*
{
    auto num_blk_src = source->get_num_blocks();
    if (source->get_block_size() == source->get_block_size()) {
                        *mem_size = source->num_bytes();
                } else if (source->get_block_size() ==
matrix::Bccoo::compression::block) { for (size_type blk_src = 0; blk_src <
num_blk_src; blk_src++) {

                                }
                } else {
                          for (size_type blk_src = 0; blk_src < num_blk_src;
blk_src++) {

                                }
                }
}
*/

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_compression(std::shared_ptr<const ReferenceExecutor> exec,
                            const matrix::Bccoo<ValueType, IndexType>* source,
                            matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COMPRESSION_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
{
    using new_precision = next_precision<ValueType>;

    size_type block_size = source->get_block_size();
    size_type num_stored_elements = source->get_num_stored_elements();

    size_type nblk_src = 0;
    size_type blk_src = 0;
    size_type row_src = 0;
    size_type col_src = 0;
    size_type shf_src = 0;
    size_type num_bytes_src = source->get_num_bytes();

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    ValueType val_src;

    size_type nblk_res = 0;
    size_type blk_res = 0;
    size_type row_res = 0;
    size_type col_res = 0;
    size_type shf_res = 0;
    size_type num_bytes_res = result->get_num_bytes();

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    new_precision val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    if (source->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data_src, offsets_data_src, nblk_src,
                                blk_src, shf_src, row_src, col_src);
            put_detect_newblock(chunk_data_res, rows_data_res, nblk_res,
                                blk_res, shf_res, row_res, row_src - row_res,
                                col_res);
            uint8 ind_src = get_position_newrow_put(
                chunk_data_src, shf_src, row_src, col_src, chunk_data_res,
                nblk_res, blk_res, rows_data_res, shf_res, row_res, col_res);
            get_next_position_value(chunk_data_src, nblk_src, ind_src, shf_src,
                                    col_src, val_src);
            val_res = val_src;
            put_next_position_value(chunk_data_res, nblk_res, col_src - col_res,
                                    shf_res, col_res, val_res);
            get_detect_endblock(block_size, nblk_src, blk_src);
            put_detect_endblock(offsets_data_res, shf_res, block_size, nblk_res,
                                blk_res);
        }
    } else {
        auto* cols_data_src = source->get_const_cols();
        auto* types_data_src = source->get_const_types();

        auto* cols_data_res = result->get_cols();
        auto* types_data_res = result->get_types();

        blk_res = blk_src = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            rows_data_res[blk_res] = rows_data_src[blk_src];
            cols_data_res[blk_res] = cols_data_src[blk_src];
            types_data_res[blk_res] = types_data_src[blk_src];
            auto type_blk = types_data_res[blk_res];
            if (type_blk & GKO_BCCOO_ROWS_MULTIPLE) {
                const uint8* rows_blk_src =
                    static_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* rows_blk_res =
                    static_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    rows_blk_res[j] = rows_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            }
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                const uint8* cols_blk_src =
                    static_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* cols_blk_res =
                    static_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    cols_blk_res[j] = cols_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                std::memcpy(
                    static_cast<unsigned char*>(chunk_data_res) + shf_res,
                    static_cast<const unsigned char*>(chunk_data_src) + shf_src,
                    block_size_local * sizeof(uint16));
                shf_src += block_size_local * sizeof(uint16);
                shf_res += block_size_local * sizeof(uint16);
            } else {
                std::memcpy(
                    static_cast<unsigned char*>(chunk_data_res) + shf_res,
                    static_cast<const unsigned char*>(chunk_data_src) + shf_src,
                    block_size_local * sizeof(uint32));
                shf_src += block_size_local * sizeof(uint32);
                shf_res += block_size_local * sizeof(uint32);
            }
            if (true) {  // TODO: ILUT table managing
                for (size_type j = 0; j < block_size_local; j++) {
                    val_src =
                        get_value_chunk<ValueType>(chunk_data_src, shf_src);
                    val_res = val_src;
                    set_value_chunk<new_precision>(chunk_data_res, shf_res,
                                                   val_res);
                    shf_src += sizeof(ValueType);
                    shf_res += sizeof(new_precision);
                }
            }
            blk_src++;
            blk_res++;
            offsets_data_res[blk_res] = shf_res;
            /*
    for (size_type j = 0; j < block_size_local; j++) {
    row_src += get_value_chunk<uint16>(chunk_data_src, shf_src);
                            shf_src += 2;
    col_src += get_value_chunk<uint16>(chunk_data_src, shf_src);
                            shf_src += 2;
    val_src += get_value_chunk<ValueType>(chunk_data_src, shf_src);
                            shf_src += sizeof(ValueType);
            set_value_chunk<uint16>(chunk_data_res, shf_res, row_src);
                            shf_res += 2;
            set_value_chunk<uint16>(chunk_data_res, shf_res, col_src);
                            shf_res += 2;
                            val_res = val_src;
            set_value_chunk<new_precision>(chunk_data_res, shf_res, val_res);
                            shf_res += sizeof(new_precision);
                    }
                    blk_src++; blk_res++;
                    offsets_data_res[blk_res] = shf_res;
            */
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
{
    size_type block_size = source->get_block_size();
    size_type num_stored_elements = source->get_num_stored_elements();

    size_type nblk = 0;
    size_type blk = 0;
    size_type row = 0;
    size_type col = 0;
    size_type shf = 0;
    size_type num_bytes = source->get_num_bytes();

    auto* rows_data = source->get_const_rows();
    auto* offsets_data = source->get_const_offsets();
    auto* chunk_data = source->get_const_chunk();
    ValueType val;

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    if (source->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row,
                                col);
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            get_detect_endblock(block_size, nblk, blk);
            row_idxs[i] = row;
            col_idxs[i] = col;
            values[i] = val;
        }
    } else {
        auto* cols_data = source->get_const_cols();
        auto* types_data = source->get_const_types();
        size_type shf_row = 0;
        size_type shf_col = 0;
        size_type shf_val = 0;

        blk = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            auto type_blk = types_data[blk];
            bool mul_row = type_blk & GKO_BCCOO_ROWS_MULTIPLE;
            bool col_8bits = type_blk & GKO_BCCOO_COLS_8BITS;
            bool col_16bits = type_blk & GKO_BCCOO_COLS_16BITS;
            row = rows_data[blk];
            col = cols_data[blk];
            shf_row = shf_col = shf;
            if (mul_row) shf_col += block_size_local;
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                shf_val = shf_col + block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                shf_val = shf_col + block_size_local * 2;
            } else {
                shf_val = shf_col + block_size_local * 4;
            }
            for (size_type j = 0; j < block_size_local; j++) {
                if (mul_row) {
                    row += get_value_chunk<uint8>(chunk_data, shf_row);
                    shf_row++;
                }
                if (col_8bits) {
                    col += get_value_chunk<uint8>(chunk_data, shf_col);
                    shf_col++;
                } else if (col_16bits) {
                    col += get_value_chunk<uint16>(chunk_data, shf_col);
                    shf_col += 2;
                } else {
                    col += get_value_chunk<uint32>(chunk_data, shf_col);
                    shf_col += 4;
                }
                val = get_value_chunk<ValueType>(chunk_data, shf_val);
                shf_val += sizeof(ValueType);
                row_idxs[i + j] = row;
                col_idxs[i + j] = col;
                values[i + j] = val;
            }
            blk++;
            shf = shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    size_type block_size = source->get_block_size();
    size_type num_stored_elements = source->get_num_stored_elements();

    size_type nblk = 0;
    size_type blk = 0;
    size_type row = 0;
    size_type col = 0;
    size_type shf = 0;
    size_type num_bytes = source->get_num_bytes();

    auto* rows_data = source->get_const_rows();
    auto* offsets_data = source->get_const_offsets();
    auto* chunk_data = source->get_const_chunk();
    ValueType val;

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    row_ptrs[0] = 0;
    if (source->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock_csr(rows_data, offsets_data, nblk, blk,
                                    row_ptrs, i, shf, row, col);
            uint8 ind =
                get_position_newrow_csr(chunk_data, row_ptrs, i, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            col_idxs[i] = col;
            values[i] = val;
            get_detect_endblock(block_size, nblk, blk);
        }
    } else {
        auto* cols_data = source->get_const_cols();
        auto* types_data = source->get_const_types();
        size_type shf_row = 0;
        size_type shf_col = 0;
        size_type shf_val = 0;
        size_type row_prv = 0;

        blk = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            auto type_blk = types_data[blk];
            bool mul_row = type_blk & GKO_BCCOO_ROWS_MULTIPLE;
            bool col_8bits = type_blk & GKO_BCCOO_COLS_8BITS;
            bool col_16bits = type_blk & GKO_BCCOO_COLS_16BITS;
            row = rows_data[blk];
            col = cols_data[blk];
            shf_row = shf_col = shf;
            if (mul_row) shf_col += block_size_local;
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                shf_val = shf_col + block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                shf_val = shf_col + block_size_local * 2;
            } else {
                shf_val = shf_col + block_size_local * 4;
            }
            for (size_type j = 0; j < block_size_local; j++) {
                if (mul_row) {
                    row += get_value_chunk<uint8>(chunk_data, shf_row);
                    shf_row++;
                }
                if (col_8bits) {
                    col += get_value_chunk<uint8>(chunk_data, shf_col);
                    shf_col++;
                } else if (col_16bits) {
                    col += get_value_chunk<uint16>(chunk_data, shf_col);
                    shf_col += 2;
                } else {
                    col += get_value_chunk<uint32>(chunk_data, shf_col);
                    shf_col += 4;
                }
                val = get_value_chunk<ValueType>(chunk_data, shf_val);
                shf_val += sizeof(ValueType);
                if (row_prv < row) row_ptrs[row] = i + j;
                col_idxs[i + j] = col;
                values[i + j] = val;
                row_prv = row;
            }
            blk++;
            shf = shf_val;
        }
    }
    if (num_stored_elements > 0) {
        row_ptrs[row + 1] = num_stored_elements;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    size_type block_size = source->get_block_size();
    size_type num_stored_elements = source->get_num_stored_elements();

    size_type nblk = 0;
    size_type blk = 0;
    size_type row = 0;
    size_type col = 0;
    size_type shf = 0;
    size_type num_bytes = source->get_num_bytes();

    auto* rows_data = source->get_const_rows();
    auto* offsets_data = source->get_const_offsets();
    auto* chunk_data = source->get_const_chunk();
    ValueType val;

    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
    }

    if (source->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row,
                                col);
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            get_detect_endblock(block_size, nblk, blk);
            result->at(row, col) += val;
        }
    } else {
        auto* cols_data = source->get_const_cols();
        auto* types_data = source->get_const_types();
        size_type shf_row = 0;
        size_type shf_col = 0;
        size_type shf_val = 0;

        blk = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            auto type_blk = types_data[blk];
            bool mul_row = type_blk & GKO_BCCOO_ROWS_MULTIPLE;
            bool col_8bits = type_blk & GKO_BCCOO_COLS_8BITS;
            bool col_16bits = type_blk & GKO_BCCOO_COLS_16BITS;
            row = rows_data[blk];
            col = cols_data[blk];
            shf_row = shf_col = shf;
            if (mul_row) shf_col += block_size_local;
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                shf_val = shf_col + block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                shf_val = shf_col + block_size_local * 2;
            } else {
                shf_val = shf_col + block_size_local * 4;
            }
            for (size_type j = 0; j < block_size_local; j++) {
                if (mul_row) {
                    row += get_value_chunk<uint8>(chunk_data, shf_row);
                    shf_row++;
                }
                if (col_8bits) {
                    col += get_value_chunk<uint8>(chunk_data, shf_col);
                    shf_col++;
                } else if (col_16bits) {
                    col += get_value_chunk<uint16>(chunk_data, shf_col);
                    shf_col += 2;
                } else {
                    col += get_value_chunk<uint32>(chunk_data, shf_col);
                    shf_col += 4;
                }
                val = get_value_chunk<ValueType>(chunk_data, shf_val);
                shf_val += sizeof(ValueType);
                result->at(row, col) += val;
            }
            blk++;
            shf = shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    size_type block_size = orig->get_block_size();
    size_type num_stored_elements = orig->get_num_stored_elements();

    size_type nblk = 0;
    size_type blk = 0;
    size_type row = 0;
    size_type col = 0;
    size_type shf = 0;
    size_type num_bytes = orig->get_num_bytes();

    auto* rows_data = orig->get_const_rows();
    auto* offsets_data = orig->get_const_offsets();
    auto* chunk_data = orig->get_const_chunk();
    auto diag_values = diag->get_values();
    ValueType val;

    auto num_rows = diag->get_size()[0];

    for (size_type row = 0; row < num_rows; row++) {
        diag_values[row] = zero<ValueType>();
    }

    if (orig->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row,
                                col);
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            get_detect_endblock(block_size, nblk, blk);
            if (row == col) {
                diag_values[row] = val;
            }
        }
    } else {
        auto* cols_data = orig->get_const_cols();
        auto* types_data = orig->get_const_types();
        size_type shf_row = 0;
        size_type shf_col = 0;
        size_type shf_val = 0;

        blk = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            auto type_blk = types_data[blk];
            bool mul_row = type_blk & GKO_BCCOO_ROWS_MULTIPLE;
            bool col_8bits = type_blk & GKO_BCCOO_COLS_8BITS;
            bool col_16bits = type_blk & GKO_BCCOO_COLS_16BITS;
            row = rows_data[blk];
            col = cols_data[blk];
            shf_row = shf_col = shf;
            if (mul_row) shf_col += block_size_local;
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                shf_val = shf_col + block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                shf_val = shf_col + block_size_local * 2;
            } else {
                shf_val = shf_col + block_size_local * 4;
            }
            for (size_type j = 0; j < block_size_local; j++) {
                if (mul_row) {
                    row += get_value_chunk<uint8>(chunk_data, shf_row);
                    shf_row++;
                }
                if (col_8bits) {
                    col += get_value_chunk<uint8>(chunk_data, shf_col);
                    shf_col++;
                } else if (col_16bits) {
                    col += get_value_chunk<uint16>(chunk_data, shf_col);
                    shf_col += 2;
                } else {
                    col += get_value_chunk<uint32>(chunk_data, shf_col);
                    shf_col += 4;
                }
                val = get_value_chunk<ValueType>(chunk_data, shf_val);
                shf_val += sizeof(ValueType);
                if (row == col) {
                    diag_values[row] = val;
                }
            }
            blk++;
            shf = shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const ReferenceExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
{
    size_type block_size = matrix->get_block_size();
    size_type num_stored_elements = matrix->get_num_stored_elements();

    size_type nblk = 0;
    size_type blk = 0;
    size_type row = 0;
    size_type col = 0;
    size_type shf = 0;
    size_type num_bytes = matrix->get_num_bytes();

    auto* rows_data = matrix->get_const_rows();
    auto* offsets_data = matrix->get_const_offsets();
    auto* chunk_data = matrix->get_chunk();
    ValueType val;

    if (matrix->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, nblk, blk, shf, row,
                                col);
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value_put(chunk_data, nblk, ind, shf, col, val,
                                        [](ValueType val) { return abs(val); });
            get_detect_endblock(block_size, nblk, blk);
        }
    } else {
        auto* cols_data = matrix->get_const_cols();
        auto* types_data = matrix->get_const_types();
        size_type shf_row = 0;
        size_type shf_col = 0;
        size_type shf_val = 0;

        blk = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            auto type_blk = types_data[blk];
            bool mul_row = type_blk & GKO_BCCOO_ROWS_MULTIPLE;
            bool col_8bits = type_blk & GKO_BCCOO_COLS_8BITS;
            bool col_16bits = type_blk & GKO_BCCOO_COLS_16BITS;
            row = rows_data[blk];
            col = cols_data[blk];
            shf_row = shf_col = shf;
            if (mul_row) shf_col += block_size_local;
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                shf_val = shf_col + block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                shf_val = shf_col + block_size_local * 2;
            } else {
                shf_val = shf_col + block_size_local * 4;
            }
            for (size_type j = 0; j < block_size_local; j++) {
                if (mul_row) {
                    row += get_value_chunk<uint8>(chunk_data, shf_row);
                    shf_row++;
                }
                if (col_8bits) {
                    col += get_value_chunk<uint8>(chunk_data, shf_col);
                    shf_col++;
                } else if (col_16bits) {
                    col += get_value_chunk<uint16>(chunk_data, shf_col);
                    shf_col += 2;
                } else {
                    col += get_value_chunk<uint32>(chunk_data, shf_col);
                    shf_col += 4;
                }
                val = get_value_chunk<ValueType>(chunk_data, shf_val);
                val = abs(val);
                set_value_chunk<ValueType>(chunk_data, shf, val);
                shf_val += sizeof(ValueType);
            }
            blk++;
            shf = shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    remove_complex<matrix::Bccoo<ValueType, IndexType>>* result)
{
    size_type block_size = source->get_block_size();
    size_type num_stored_elements = source->get_num_stored_elements();

    size_type nblk_src = 0;
    size_type blk_src = 0;
    size_type row_src = 0;
    size_type col_src = 0;
    size_type shf_src = 0;
    size_type num_bytes_src = source->get_num_bytes();

    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    ValueType val_src;

    size_type nblk_res = 0;
    size_type blk_res = 0;
    size_type row_res = 0;
    size_type col_res = 0;
    size_type shf_res = 0;
    size_type num_bytes_res = result->get_num_bytes();

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    remove_complex<ValueType> val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    if (source->use_block_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data_src, offsets_data_src, nblk_src,
                                blk_src, shf_src, row_src, col_src);
            put_detect_newblock(chunk_data_res, rows_data_res, nblk_res,
                                blk_res, shf_res, row_res, row_src - row_res,
                                col_res);
            uint8 ind_src = get_position_newrow_put(
                chunk_data_src, shf_src, row_src, col_src, chunk_data_res,
                nblk_res, blk_res, rows_data_res, shf_res, row_res, col_res);
            get_next_position_value(chunk_data_src, nblk_src, ind_src, shf_src,
                                    col_src, val_src);
            val_res = abs(val_src);
            put_next_position_value(chunk_data_res, nblk_res, col_src - col_res,
                                    shf_res, col_res, val_res);
            get_detect_endblock(block_size, nblk_src, blk_src);
            put_detect_endblock(offsets_data_res, shf_res, block_size, nblk_res,
                                blk_res);
        }
    } else {
        auto* cols_data_src = source->get_const_cols();
        auto* types_data_src = source->get_const_types();

        auto* cols_data_res = result->get_cols();
        auto* types_data_res = result->get_types();

        blk_res = blk_src = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local = num_stored_elements - i;
            rows_data_res[blk_res] = rows_data_src[blk_src];
            cols_data_res[blk_res] = cols_data_src[blk_src];
            types_data_res[blk_res] = types_data_src[blk_src];
            auto type_blk = types_data_res[blk_res];
            if (type_blk & GKO_BCCOO_ROWS_MULTIPLE) {
                const uint8* rows_blk_src =
                    static_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* rows_blk_res =
                    static_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    rows_blk_res[j] = rows_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            }
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                const uint8* cols_blk_src =
                    static_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* cols_blk_res =
                    static_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    cols_blk_res[j] = cols_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                std::memcpy(
                    static_cast<unsigned char*>(chunk_data_res) + shf_res,
                    static_cast<const unsigned char*>(chunk_data_src) + shf_src,
                    block_size_local * sizeof(uint16));
                shf_src += block_size_local * sizeof(uint16);
                shf_res += block_size_local * sizeof(uint16);
            } else {
                std::memcpy(
                    static_cast<unsigned char*>(chunk_data_res) + shf_res,
                    static_cast<const unsigned char*>(chunk_data_src) + shf_src,
                    block_size_local * sizeof(uint32));
                shf_src += block_size_local * sizeof(uint32);
                shf_res += block_size_local * sizeof(uint32);
            }
            if (true) {
                for (size_type j = 0; j < block_size_local; j++) {
                    val_src =
                        get_value_chunk<ValueType>(chunk_data_src, shf_src);
                    val_res = abs(val_src);
                    set_value_chunk<ValueType>(chunk_data_res, shf_res,
                                               val_res);
                    shf_src += sizeof(ValueType);
                    shf_res += sizeof(ValueType);
                }
            }
            blk_src++;
            blk_res++;
            offsets_data_res[blk_res] = shf_res;
            /*
    for (size_type j = 0; j < block_size_local; j++) {
    row_src += get_value_chunk<uint16>(chunk_data_src, shf_src);
                            shf_src += 2;
    col_src += get_value_chunk<uint16>(chunk_data_src, shf_src);
                            shf_src += 2;
    val_src += get_value_chunk<ValueType>(chunk_data_src, shf_src);
                            shf_src += sizeof(ValueType);
            set_value_chunk<uint16>(chunk_data_res, shf_res, row_src);
                            shf_res += 2;
            set_value_chunk<uint16>(chunk_data_res, shf_res, col_src);
                            shf_res += 2;
                            val_res = val_src;
            set_value_chunk<new_precision>(chunk_data_res, shf_res, val_res);
                            shf_res += sizeof(new_precision);
                    }
                    blk_src++; blk_res++;
                    offsets_data_res[blk_res] = shf_res;
            */
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace reference
}  // namespace kernels
}  // namespace gko
