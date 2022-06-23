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


#include "core/base/unaligned_access.hpp"
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


void get_default_block_size(std::shared_ptr<const ReferenceExecutor> exec,
                            size_type* block_size)
{
    *block_size = 10;
}


void get_default_compression(std::shared_ptr<const ReferenceExecutor> exec,
                             matrix::bccoo::compression* compression)
{
    *compression = matrix::bccoo::compression::element;
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

    auto block_size = a->get_block_size();
    auto num_stored_elements = a->get_num_stored_elements();
    auto num_cols = b->get_size()[1];

    compr_idxs idxs = {};
    ValueType val;

    // Computation of chunk
    if (a->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
            for (size_type j = 0; j < num_cols; j++) {
                c->at(idxs.row, j) += val * b->at(idxs.col, j);
            }
        }
    } else {
        auto* cols_data = a->get_const_cols();
        auto* types_data = a->get_const_types();

        compr_blk_idxs blk_idxs = {};

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Counting bytes to write (row,col,val) on result
                for (size_type k = 0; k < num_cols; k++) {
                    c->at(idxs.row, k) += val * b->at(idxs.col, k);
                }
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
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

    compr_idxs idxs = {};
    ValueType val;

    // Computation of chunk
    if (a->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
            for (size_type j = 0; j < num_cols; j++) {
                c->at(idxs.row, j) += alpha_val * val * b->at(idxs.col, j);
            }
        }
    } else {
        auto* cols_data = a->get_const_cols();
        auto* types_data = a->get_const_types();

        compr_blk_idxs blk_idxs = {};

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Counting bytes to write (row,col,val) on result
                for (size_type k = 0; k < num_cols; k++) {
                    c->at(idxs.row, k) += alpha_val * val * b->at(idxs.col, k);
                }
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_elm_elm(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::bccoo::compression compress_res, const size_type block_size_res,
    size_type* mem_size)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType val_src;
    compr_idxs idxs_res = {};
    ValueType val_res;
    for (size_type i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                            idxs_src.blk, idxs_src.shf, idxs_src.row,
                            idxs_src.col);
        uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                            idxs_src.row, idxs_src.col);
        get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                idxs_src.shf, idxs_src.col, val_src);
        get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
        // Counting bytes to write (row,col,val) on result
        cnt_detect_newblock(idxs_res.nblk, idxs_res.shf, idxs_res.row,
                            idxs_src.row - idxs_res.row, idxs_res.col);
        size_type col_src_res = cnt_position_newrow_mat_data(
            idxs_src.row, idxs_src.col, idxs_res.shf, idxs_res.row,
            idxs_res.col);
        cnt_next_position_value(col_src_res, idxs_res.shf, idxs_res.col,
                                val_src, idxs_res.nblk);
        cnt_detect_endblock(block_size_res, idxs_res.nblk, idxs_res.blk);
    }
    *mem_size = idxs_res.shf;
}


template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_elm_blk(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::bccoo::compression compress_res, const size_type block_size_res,
    size_type* mem_size)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType val_src;

    compr_idxs idxs_res = {};

    for (size_type i = 0; i < num_stored_elements; i += block_size_res) {
        size_type block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_blk_idxs blk_idxs_res = {};
        blk_idxs_res.row_frs = idxs_src.row;
        blk_idxs_res.col_frs = idxs_src.col;
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                                idxs_src.blk, idxs_src.shf, idxs_src.row,
                                idxs_src.col);
            uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                                idxs_src.row, idxs_src.col);
            get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                    idxs_src.shf, idxs_src.col, val_src);
            get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
            // Analyzing the impact of (row,col,val) in the block
            blk_idxs_res.mul_row =
                blk_idxs_res.mul_row || (idxs_src.row != blk_idxs_res.row_frs);
            if (idxs_src.col < blk_idxs_res.col_frs) {
                blk_idxs_res.col_dif += (blk_idxs_res.col_frs - idxs_src.col);
                blk_idxs_res.col_frs = idxs_src.col;
            } else if (idxs_src.col >
                       (blk_idxs_res.col_frs + blk_idxs_res.col_dif)) {
                blk_idxs_res.col_dif = idxs_src.col - blk_idxs_res.col_frs;
            }
        }
        // Counting bytes to write block on result
        if (blk_idxs_res.mul_row) idxs_res.shf += block_size_local;
        if (blk_idxs_res.col_dif <= 0xFF) {
            idxs_res.shf += block_size_local;
        } else if (blk_idxs_res.col_dif <= 0xFFFF) {
            idxs_res.shf += block_size_local * sizeof(uint16);
        } else {
            idxs_res.shf += block_size_local * sizeof(uint32);
        }
        idxs_res.shf += block_size_local * sizeof(ValueType);
    }
    *mem_size = idxs_res.shf;
}


template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_blk_elm(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::bccoo::compression compress_res, const size_type block_size_res,
    size_type* mem_size)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    auto block_size_src = source->get_block_size();
    auto num_bytes_src = source->get_num_bytes();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType val_src;

    compr_idxs idxs_res = {};

    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local =
            std::min(block_size_src, num_stored_elements - i);
        init_block_indices(rows_data_src, cols_data_src, block_size_local,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            // Counting bytes to write (row,col,val) on result
            cnt_detect_newblock(idxs_res.nblk, idxs_res.shf, idxs_res.row,
                                idxs_src.row - idxs_res.row, idxs_res.col);
            size_type col_src_res = cnt_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, idxs_res.shf, idxs_res.row,
                idxs_res.col);
            cnt_next_position_value(col_src_res, idxs_res.shf, idxs_res.col,
                                    val_src, idxs_res.nblk);
            cnt_detect_endblock(block_size_res, idxs_res.nblk, idxs_res.blk);
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


template <typename ValueType, typename IndexType>
inline void mem_size_bccoo_blk_blk(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::bccoo::compression compress_res, const size_type block_size_res,
    size_type* mem_size)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    auto block_size_src = source->get_block_size();
    auto num_bytes_src = source->get_num_bytes();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType val_src;

    auto* rows_data_res = source->get_const_rows();
    auto* offsets_data_res = source->get_const_offsets();
    auto* chunk_data_res = source->get_const_chunk();
    auto* cols_data_res = source->get_const_cols();
    auto* types_data_res = source->get_const_types();

    compr_idxs idxs_res = {};
    compr_blk_idxs blk_idxs_res = {};
    ValueType val_res;

    size_type i_res = 0;
    size_type j_res = 0;
    size_type block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        init_block_indices(rows_data_src, cols_data_src, block_size_local_src,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            if (j_res == 0) {
                blk_idxs_res.row_frs = idxs_src.row;
                blk_idxs_res.col_frs = idxs_src.col;
                blk_idxs_res.col_dif = 0;
            }
            blk_idxs_res.mul_row =
                blk_idxs_res.mul_row || (idxs_src.row != blk_idxs_res.row_frs);
            if (idxs_src.col < blk_idxs_res.col_frs) {
                blk_idxs_res.col_dif += (blk_idxs_res.col_frs - idxs_src.col);
                blk_idxs_res.col_frs = idxs_src.col;
            } else if (idxs_src.col >
                       (blk_idxs_res.col_frs + blk_idxs_res.col_dif)) {
                blk_idxs_res.col_dif = idxs_src.col - blk_idxs_res.col_frs;
            }
            j_res++;
            if (j_res == block_size_local_res) {
                // Counting bytes to write block on result
                if (blk_idxs_res.mul_row) {
                    idxs_res.shf += block_size_local_res;
                }
                if (blk_idxs_res.col_dif <= 0xFF) {
                    idxs_res.shf += block_size_local_res;
                } else if (blk_idxs_res.col_dif <= 0xFFFF) {
                    idxs_res.shf += block_size_local_res * sizeof(uint16);
                } else {
                    idxs_res.shf += block_size_local_res * sizeof(uint32);
                }
                idxs_res.shf += sizeof(ValueType) * block_size_local_res;
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                j_res = 0;
            }
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    *mem_size = idxs_res.shf;
}


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::bccoo::compression compress_res,
                    const size_type block_size_res, size_type* mem_size)
{
    if ((source->get_block_size() == block_size_res) &&
        (source->get_compression() == compress_res)) {
        *mem_size = source->get_num_bytes();
    } else if ((source->use_element_compression()) &&
               (compress_res == source->get_compression())) {
        mem_size_bccoo_elm_elm(exec, source, compress_res, block_size_res,
                               mem_size);
    } else if (source->use_element_compression()) {
        mem_size_bccoo_elm_blk(exec, source, compress_res, block_size_res,
                               mem_size);
    } else if (compress_res == matrix::bccoo::compression::element) {
        mem_size_bccoo_blk_elm(exec, source, compress_res, block_size_res,
                               mem_size);
    } else {
        mem_size_bccoo_blk_blk(exec, source, compress_res, block_size_res,
                               mem_size);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_bccoo_copy(std::shared_ptr<const ReferenceExecutor> exec,
                           const matrix::Bccoo<ValueType, IndexType>* source,
                           matrix::Bccoo<ValueType, IndexType>* result)
{
    // Try to remmove static_cast
    if (source->get_num_stored_elements() > 0) {
        if (source->use_element_compression()) {
            //            std::memcpy(static_cast<IndexType*>(result->get_rows()),
            std::memcpy(
                (result->get_rows()), (source->get_const_rows()),
                //                        static_cast<const
                //                        IndexType*>(source->get_const_rows()),
                source->get_num_blocks() * sizeof(IndexType));
            auto offsets_data_src = source->get_const_offsets();
            auto offsets_data_res = result->get_offsets();
            //            std::memcpy(static_cast<IndexType*>(offsets_data_res),
            std::memcpy((offsets_data_res), (offsets_data_src),
                        //                        static_cast<const
                        //                        IndexType*>(offsets_data_src),
                        (source->get_num_blocks() + 1) * sizeof(IndexType));
            auto chunk_data_src = source->get_const_chunk();
            auto chunk_data_res = result->get_chunk();
            //            std::memcpy(static_cast<uint8*>(chunk_data_res),
            std::memcpy((chunk_data_res), (chunk_data_src),
                        //                        static_cast<const
                        //                        uint8*>(chunk_data_src),
                        source->get_num_bytes() * sizeof(uint8));
        } else {
            //            std::memcpy(static_cast<IndexType*>(result->get_rows()),
            std::memcpy(
                (result->get_rows()), (source->get_const_rows()),
                //                        static_cast<const
                //                        IndexType*>(source->get_const_rows()),
                source->get_num_blocks() * sizeof(IndexType));
            //            std::memcpy(static_cast<IndexType*>(result->get_cols()),
            std::memcpy(
                (result->get_cols()), (source->get_const_cols()),
                //                        static_cast<const
                //                        IndexType*>(source->get_const_cols()),
                source->get_num_blocks() * sizeof(IndexType));
            //            std::memcpy(static_cast<uint8*>(result->get_types()),
            std::memcpy(
                (result->get_types()), (source->get_const_types()),
                //                        static_cast<const
                //                        uint8*>(source->get_const_types()),
                source->get_num_blocks() * sizeof(uint8));
            std::memcpy(
                //                static_cast<IndexType*>(result->get_offsets()),
                (result->get_offsets()), (source->get_const_offsets()),
                //                static_cast<const
                //                IndexType*>(source->get_const_offsets()),
                (source->get_num_blocks() + 1) * sizeof(IndexType));
            //            std::memcpy(static_cast<uint8*>(result->get_chunk()),
            std::memcpy(
                (result->get_chunk()), (source->get_const_chunk()),
                //                        static_cast<const
                //                        uint8*>(source->get_const_chunk()),
                source->get_num_bytes() * sizeof(uint8));
        }
    }
}


template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_elm_elm(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();

    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i++) {
        // Reading (row,col,val) from source
        get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                            idxs_src.blk, idxs_src.shf, idxs_src.row,
                            idxs_src.col);
        uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                            idxs_src.row, idxs_src.col);
        get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                idxs_src.shf, idxs_src.col, val_src);
        get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
        // Writing (row,col,val) to result
        val_res = finalize_op(val_src);
        put_detect_newblock(rows_data_res, idxs_res.nblk, idxs_res.blk,
                            idxs_res.row, idxs_src.row - idxs_res.row,
                            idxs_res.col);
        size_type col_src_res = put_position_newrow_mat_data(
            idxs_src.row, idxs_src.col, chunk_data_res, idxs_res.shf,
            idxs_res.row, idxs_res.col);
        put_next_position_value(chunk_data_res, idxs_res.nblk, col_src_res,
                                idxs_res.shf, idxs_res.col, val_res);
        put_detect_endblock(offsets_data_res, idxs_res.shf, block_size_res,
                            idxs_res.nblk, idxs_res.blk);
    }
    if (idxs_res.nblk > 0) {
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_elm_blk(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();

    auto num_stored_elements = source->get_num_stored_elements();
    auto block_size_src = source->get_block_size();

    compr_idxs idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    auto* cols_data_res = result->get_cols();
    auto* types_data_res = result->get_types();

    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    ValueType_res val_res;

    array<IndexType> rows_blk(exec, block_size_res);
    array<IndexType> cols_blk(exec, block_size_res);
    array<ValueType_res> vals_blk(exec, block_size_res);

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i += block_size_res) {
        size_type block_size_local =
            std::min(block_size_res, num_stored_elements - i);
        compr_blk_idxs blk_idxs_res = {};
        uint8 type_blk = {};

        blk_idxs_res.row_frs = idxs_src.row;
        blk_idxs_res.col_frs = idxs_src.col;
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_detect_newblock(rows_data_src, offsets_data_src, idxs_src.nblk,
                                idxs_src.blk, idxs_src.shf, idxs_src.row,
                                idxs_src.col);
            uint8 ind_src = get_position_newrow(chunk_data_src, idxs_src.shf,
                                                idxs_src.row, idxs_src.col);
            get_next_position_value(chunk_data_src, idxs_src.nblk, ind_src,
                                    idxs_src.shf, idxs_src.col, val_src);
            get_detect_endblock(block_size_src, idxs_src.nblk, idxs_src.blk);
            // Analyzing the impact of (row,col,val) in the block
            if (j == 0) {
                blk_idxs_res.row_frs = idxs_src.row;
                blk_idxs_res.col_frs = idxs_src.col;
                blk_idxs_res.col_dif = 0;
            }
            rows_blk.get_data()[j] = idxs_src.row;
            cols_blk.get_data()[j] = idxs_src.col;
            vals_blk.get_data()[j] = val_src;
            blk_idxs_res.mul_row =
                blk_idxs_res.mul_row || (idxs_src.row != blk_idxs_res.row_frs);
            if (idxs_src.col < blk_idxs_res.col_frs) {
                blk_idxs_res.col_dif += (blk_idxs_res.col_frs - idxs_src.col);
                blk_idxs_res.col_frs = idxs_src.col;
            } else if (idxs_src.col >
                       (blk_idxs_res.col_frs + blk_idxs_res.col_dif)) {
                blk_idxs_res.col_dif = idxs_src.col - blk_idxs_res.col_frs;
            }
        }
        // Counting bytes to write block on result
        idxs_res.nblk = block_size_local;
        type_blk = generate_type_blk(idxs_res, blk_idxs_res, rows_blk, cols_blk,
                                     vals_blk, chunk_data_res);
        rows_data_res[idxs_res.blk] = blk_idxs_res.row_frs;
        cols_data_res[idxs_res.blk] = blk_idxs_res.col_frs;
        types_data_res[idxs_res.blk] = type_blk;
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


template <typename ValueType_src, typename ValueType_res, typename IndexType>
void convert_to_bccoo_blk_elm(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    size_type block_size_src = source->get_block_size();
    size_type num_bytes_src = source->get_num_bytes();
    size_type num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    ValueType_res val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local =
            std::min(block_size_src, num_stored_elements - i);

        init_block_indices(rows_data_src, cols_data_src, block_size_local,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType_src>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            // Writing (row,col,val) to result
            val_res = val_src;
            put_detect_newblock(rows_data_res, idxs_res.nblk, idxs_res.blk,
                                idxs_res.row, idxs_src.row - idxs_res.row,
                                idxs_res.col);
            size_type col_src_res = put_position_newrow_mat_data(
                idxs_src.row, idxs_src.col, chunk_data_res, idxs_res.shf,
                idxs_res.row, idxs_res.col);
            put_next_position_value(chunk_data_res, idxs_res.nblk, col_src_res,
                                    idxs_res.shf, idxs_res.col, val_res);
            put_detect_endblock(offsets_data_res, idxs_res.shf, block_size_res,
                                idxs_res.nblk, idxs_res.blk);
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
    if (idxs_res.nblk > 0) {
        offsets_data_res[++idxs_res.blk] = idxs_res.shf;
    }
}


template <typename ValueType_src, typename ValueType_res, typename IndexType,
          typename Callable>
void convert_to_bccoo_blk_blk(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType_src, IndexType>* source,
    matrix::Bccoo<ValueType_res, IndexType>* result, Callable finalize_op)
{
    auto* rows_data_src = source->get_const_rows();
    auto* offsets_data_src = source->get_const_offsets();
    auto* chunk_data_src = source->get_const_chunk();
    auto* cols_data_src = source->get_const_cols();
    auto* types_data_src = source->get_const_types();

    auto block_size_src = source->get_block_size();
    auto num_bytes_src = source->get_num_bytes();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs_src = {};
    compr_blk_idxs blk_idxs_src = {};
    ValueType_src val_src;

    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    auto* cols_data_res = result->get_cols();
    auto* types_data_res = result->get_types();

    auto block_size_res = result->get_block_size();

    compr_idxs idxs_res = {};
    compr_blk_idxs blk_idxs_res = {};

    array<IndexType> rows_blk_res(exec, block_size_res);
    array<IndexType> cols_blk_res(exec, block_size_res);
    array<ValueType_res> vals_blk_res(exec, block_size_res);

    uint8 type_blk = {};
    size_type i_res = 0;
    size_type j_res = 0;
    size_type block_size_local_res =
        std::min(block_size_res, num_stored_elements - i_res);

    blk_idxs_res.row_frs = idxs_src.row;
    blk_idxs_res.col_frs = idxs_src.col;
    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
        size_type block_size_local_src =
            std::min(block_size_src, num_stored_elements - i);
        init_block_indices(rows_data_src, cols_data_src, block_size_local_src,
                           idxs_src, types_data_src[idxs_src.blk],
                           blk_idxs_src);
        for (size_type j = 0; j < block_size_local_src; j++) {
            // Reading (row,col,val) from source
            get_block_position_value<IndexType, ValueType_src>(
                chunk_data_src, blk_idxs_src, idxs_src.row, idxs_src.col,
                val_src);
            // Analyzing the impact of (row,col,val) in the block
            if (j_res == 0) {
                blk_idxs_res.row_frs = idxs_src.row;
                blk_idxs_res.col_frs = idxs_src.col;
                blk_idxs_res.col_dif = 0;
                blk_idxs_res.mul_row = false;
            }
            rows_blk_res.get_data()[j_res] = idxs_src.row;
            cols_blk_res.get_data()[j_res] = idxs_src.col;
            vals_blk_res.get_data()[j_res] =
                (ValueType_res)finalize_op(val_src);
            blk_idxs_res.mul_row =
                blk_idxs_res.mul_row || (idxs_src.row != blk_idxs_res.row_frs);
            if (idxs_src.col < blk_idxs_res.col_frs) {
                blk_idxs_res.col_dif += (blk_idxs_res.col_frs - idxs_src.col);
                blk_idxs_res.col_frs = idxs_src.col;
            } else if (idxs_src.col >
                       (blk_idxs_res.col_frs + blk_idxs_res.col_dif)) {
                blk_idxs_res.col_dif = idxs_src.col - blk_idxs_res.col_frs;
            }
            j_res++;
            if (j_res == block_size_local_res) {
                // Counting bytes to write block on result
                idxs_res.nblk = block_size_local_res;
                type_blk = generate_type_blk(idxs_res, blk_idxs_res,
                                             rows_blk_res, cols_blk_res,
                                             vals_blk_res, chunk_data_res);
                rows_data_res[idxs_res.blk] = blk_idxs_res.row_frs;
                cols_data_res[idxs_res.blk] = blk_idxs_res.col_frs;
                types_data_res[idxs_res.blk] = type_blk;
                offsets_data_res[++idxs_res.blk] = idxs_res.shf;
                i_res += block_size_local_res;
                block_size_local_res =
                    std::min(block_size_res, num_stored_elements - i_res);
                j_res = 0;
            }
        }
        idxs_src.blk++;
        idxs_src.shf = blk_idxs_src.shf_val;
    }
}


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
{
    auto block_size_res = result->get_block_size();
    auto compress_res = result->get_compression();
    if ((source->get_block_size() == result->get_block_size()) &&
        (source->get_compression() == result->get_compression())) {
        convert_to_bccoo_copy(exec, source, result);
    } else if ((source->use_element_compression()) &&
               (result->use_element_compression())) {
        convert_to_bccoo_elm_elm(exec, source, result,
                                 [](ValueType val) { return val; });
    } else if (source->use_element_compression()) {
        convert_to_bccoo_elm_blk(exec, source, result);
    } else if (compress_res == matrix::bccoo::compression::element) {
        convert_to_bccoo_blk_elm(exec, source, result);
    } else {
        convert_to_bccoo_blk_blk(exec, source, result,
                                 [](ValueType val) { return val; });
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_BCCOO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
{
    //    using new_precision = next_precision<ValueType>;
    //    auto block_size_res = result->get_block_size();
    auto compress_res = result->get_compression();
    //    if ((source->get_block_size() == result->get_block_size()) &&
    //        (source->get_compression() == result->get_compression())) {
    //        convert_to_bccoo_copy(exec, source, result);
    //    } else
    if ((source->use_element_compression()) &&
        (result->use_element_compression())) {
        convert_to_bccoo_elm_elm(exec, source, result,
                                 [](ValueType val) { return val; });
    } else if (source->use_element_compression()) {
        convert_to_bccoo_elm_blk(exec, source, result);
    } else if (compress_res == matrix::bccoo::compression::element) {
        convert_to_bccoo_blk_elm(exec, source, result);
    } else {
        convert_to_bccoo_blk_blk(exec, source, result,
                                 [](ValueType val) { return val; });
    }
}
/*
{  // TODO: allow the use of different block_size in source and result
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
    if (source->use_element_compression()) {
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
        //	std::cout << "CONVERT BLOCK NEXT PRECISION" std::endl;
        auto* cols_data_src = source->get_const_cols();
        auto* types_data_src = source->get_const_types();

        auto* cols_data_res = result->get_cols();
        auto* types_data_res = result->get_types();

        blk_res = blk_src = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            // std::cout << blk_src << " - " << shf_src << " - "
            //					<< shf_res << std::endl;
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            rows_data_res[blk_res] = rows_data_src[blk_src];
            cols_data_res[blk_res] = cols_data_src[blk_src];
            types_data_res[blk_res] = types_data_src[blk_src];
            auto type_blk = types_data_res[blk_res];
            if (type_blk & GKO_BCCOO_ROWS_MULTIPLE) {
                // std::cout << "MULTIROWS" << std::endl;
                const uint8* rows_blk_src =
                    reinterpret_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* rows_blk_res =
                    reinterpret_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    rows_blk_res[j] = rows_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            }
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                //	std::cout << "COL8BITS" << std::endl;
                const uint8* cols_blk_src =
                    reinterpret_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* cols_blk_res =
                    reinterpret_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    cols_blk_res[j] = cols_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                //	std::cout << "COL16BITS" << std::endl;
                std::memcpy(
                    reinterpret_cast<uint16*>(chunk_data_res + shf_res),
                    reinterpret_cast<const uint16*>(chunk_data_src + shf_src),
                    block_size_local * sizeof(uint16));
                shf_src += block_size_local * sizeof(uint16);
                shf_res += block_size_local * sizeof(uint16);
            } else {
                //	std::cout << "COL32BITS" << std::endl;
                std::memcpy(
                    reinterpret_cast<uint32*>(chunk_data_res + shf_res),
                    reinterpret_cast<const uint32*>(chunk_data_src + shf_src),
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
        }
        //	std::cout << blk_src << " - " << shf_src
        //						<< " - " << shf_res <<
        // std::endl;
    }
}
*/

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
{
    auto* rows_data = source->get_const_rows();
    auto* offsets_data = source->get_const_offsets();
    auto* chunk_data = source->get_const_chunk();

    auto block_size = source->get_block_size();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs = {};
    ValueType val;

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    if (source->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from matrix
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
            // Writing (row,col,val) on result
            row_idxs[i] = idxs.row;
            col_idxs[i] = idxs.col;
            values[i] = val;
        }
    } else {
        auto* cols_data = source->get_const_cols();
        auto* types_data = source->get_const_types();

        compr_blk_idxs blk_idxs = {};

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Writing (row,col,val) on result
                row_idxs[i + j] = idxs.row;
                col_idxs[i + j] = idxs.col;
                values[i + j] = val;
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
}
/*
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

    if (source->use_element_compression()) {
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
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            size_type row_frs;
            size_type col_frs;
            bool mul_row;
            bool col_8bits;
            bool col_16bits;
            init_block_indices(rows_data, cols_data, block_size_local, blk, shf,
                               types_data[blk], mul_row, col_8bits, col_16bits,
                               row_frs, col_frs, shf_row, shf_col, shf_val);
            for (size_type j = 0; j < block_size_local; j++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, mul_row, col_8bits, col_16bits, row_frs,
                    col_frs, row, col, val, shf_row, shf_col, shf_val);
                row_idxs[i + j] = row;
                col_idxs[i + j] = col;
                values[i + j] = val;
            }
            blk++;
            shf = shf_val;
        }
    }
}
*/
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto* rows_data = source->get_const_rows();
    auto* offsets_data = source->get_const_offsets();
    auto* chunk_data = source->get_const_chunk();

    auto block_size = source->get_block_size();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs = {};
    ValueType val;

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    row_ptrs[0] = 0;
    if (source->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from matrix
            get_detect_newblock_csr(rows_data, offsets_data, idxs.nblk,
                                    idxs.blk, row_ptrs, i, idxs.shf, idxs.row,
                                    idxs.col);
            uint8 ind = get_position_newrow_csr(chunk_data, row_ptrs, i,
                                                idxs.shf, idxs.row, idxs.col);
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
            // Writing (row,col,val) on result
            col_idxs[i] = idxs.col;
            values[i] = val;
        }
    } else {
        auto* cols_data = source->get_const_cols();
        auto* types_data = source->get_const_types();

        compr_blk_idxs blk_idxs = {};
        size_type row_prv = 0;

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Writing (row,col,val) on result
                if (row_prv < idxs.row) row_ptrs[idxs.row] = i + j;
                col_idxs[i + j] = idxs.col;
                values[i + j] = val;
                row_prv = idxs.row;
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
    if (num_stored_elements > 0) {
        row_ptrs[idxs.row + 1] = num_stored_elements;
    }
}
/*
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
    if (source->use_element_compression()) {
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
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            size_type row_frs;
            size_type col_frs;
            bool mul_row;
            bool col_8bits;
            bool col_16bits;
            init_block_indices(rows_data, cols_data, block_size_local, blk, shf,
                               types_data[blk], mul_row, col_8bits, col_16bits,
                               row_frs, col_frs, shf_row, shf_col, shf_val);
            for (size_type j = 0; j < block_size_local; j++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, mul_row, col_8bits, col_16bits, row_frs,
                    col_frs, row, col, val, shf_row, shf_col, shf_val);
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
*/

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    auto* rows_data = source->get_const_rows();
    auto* offsets_data = source->get_const_offsets();
    auto* chunk_data = source->get_const_chunk();

    auto block_size = source->get_block_size();
    auto num_stored_elements = source->get_num_stored_elements();

    compr_idxs idxs = {};
    ValueType val;

    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
    }

    if (source->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from matrix
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
            // Writing (row,col,val) on result
            result->at(idxs.row, idxs.col) += val;
        }
    } else {
        auto* cols_data = source->get_const_cols();
        auto* types_data = source->get_const_types();

        compr_blk_idxs blk_idxs = {};

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Writing (row,col,val) on result
                result->at(idxs.row, idxs.col) += val;
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
}
/*
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

    if (source->use_element_compression()) {
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
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            size_type row_frs;
            size_type col_frs;
            bool mul_row;
            bool col_8bits;
            bool col_16bits;
            init_block_indices(rows_data, cols_data, block_size_local, blk, shf,
                               types_data[blk], mul_row, col_8bits, col_16bits,
                               row_frs, col_frs, shf_row, shf_col, shf_val);
            for (size_type j = 0; j < block_size_local; j++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, mul_row, col_8bits, col_16bits, row_frs,
                    col_frs, row, col, val, shf_row, shf_col, shf_val);
                result->at(row, col) += val;
            }
            blk++;
            shf = shf_val;
        }
    }
}
*/

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    auto* rows_data = orig->get_const_rows();
    auto* offsets_data = orig->get_const_offsets();
    auto* chunk_data = orig->get_const_chunk();

    auto block_size = orig->get_block_size();
    auto num_stored_elements = orig->get_num_stored_elements();

    compr_idxs idxs = {};
    ValueType val;

    auto diag_values = diag->get_values();
    auto num_rows = diag->get_size()[0];
    for (size_type row = 0; row < num_rows; row++) {
        diag_values[row] = zero<ValueType>();
    }

    if (orig->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            // Reading (row,col,val) from matrix
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            get_next_position_value(chunk_data, idxs.nblk, ind, idxs.shf,
                                    idxs.col, val);
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
            // Writing (row,col,val) on result
            if (idxs.row == idxs.col) {
                diag_values[idxs.row] = val;
            }
        }
    } else {
        auto* cols_data = orig->get_const_cols();
        auto* types_data = orig->get_const_types();

        compr_blk_idxs blk_idxs = {};

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading (row,col,val) from matrix
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val);
                // Writing (row,col,val) on result
                if (idxs.row == idxs.col) {
                    diag_values[idxs.row] = val;
                }
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
}
/*
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

    if (orig->use_element_compression()) {
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
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            size_type row_frs;
            size_type col_frs;
            bool mul_row;
            bool col_8bits;
            bool col_16bits;
            init_block_indices(rows_data, cols_data, block_size_local, blk, shf,
                               types_data[blk], mul_row, col_8bits, col_16bits,
                               row_frs, col_frs, shf_row, shf_col, shf_val);
            for (size_type j = 0; j < block_size_local; j++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, mul_row, col_8bits, col_16bits, row_frs,
                    col_frs, row, col, val, shf_row, shf_col, shf_val);
                if (row == col) {
                    diag_values[row] = val;
                }
            }
            blk++;
            shf = shf_val;
        }
    }
}
*/
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const ReferenceExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
{
    auto* rows_data = matrix->get_const_rows();
    auto* offsets_data = matrix->get_const_offsets();
    auto* chunk_data = matrix->get_chunk();

    auto block_size = matrix->get_block_size();
    auto num_stored_elements = matrix->get_num_stored_elements();

    compr_idxs idxs = {};
    ValueType val;

    if (matrix->use_element_compression()) {
        for (size_type i = 0; i < num_stored_elements; i++) {
            // Reading/Writing (row,col,val) from matrix
            get_detect_newblock(rows_data, offsets_data, idxs.nblk, idxs.blk,
                                idxs.shf, idxs.row, idxs.col);
            uint8 ind =
                get_position_newrow(chunk_data, idxs.shf, idxs.row, idxs.col);
            get_next_position_value_put(chunk_data, idxs.nblk, ind, idxs.shf,
                                        idxs.col, val,
                                        [](ValueType val) { return abs(val); });
            get_detect_endblock(block_size, idxs.nblk, idxs.blk);
        }
    } else {
        auto* cols_data = matrix->get_const_cols();
        auto* types_data = matrix->get_const_types();

        compr_blk_idxs blk_idxs = {};

        for (size_type i = 0; i < num_stored_elements; i += block_size) {
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            init_block_indices(rows_data, cols_data, block_size_local, idxs,
                               types_data[idxs.blk], blk_idxs);
            for (size_type j = 0; j < block_size_local; j++) {
                // Reading/Writing (row,col,val) from matrix
                get_block_position_value_put<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs.row, idxs.col, val,
                    [](ValueType val) { return abs(val); });
            }
            idxs.blk++;
            idxs.shf = blk_idxs.shf_val;
        }
    }
}
/*
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

    if (matrix->use_element_compression()) {
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
            size_type block_size_local =
                std::min(block_size, num_stored_elements - i);
            size_type row_frs;
            size_type col_frs;
            bool mul_row;
            bool col_8bits;
            bool col_16bits;
            init_block_indices(rows_data, cols_data, block_size_local, blk, shf,
                               types_data[blk], mul_row, col_8bits, col_16bits,
                               row_frs, col_frs, shf_row, shf_col, shf_val);
            for (size_type j = 0; j < block_size_local; j++) {
                get_block_position_value_put<IndexType, ValueType>(
                    chunk_data, mul_row, col_8bits, col_16bits, row_frs,
                    col_frs, row, col, val, shf_row, shf_col, shf_val,
                    [](ValueType val) { return abs(val); });
            }
            blk++;
            shf = shf_val;
        }
    }
}
*/
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    remove_complex<matrix::Bccoo<ValueType, IndexType>>* result)
/**/
{
    if (source->use_element_compression()) {
        convert_to_bccoo_elm_elm(exec, source, result,
                                 [](ValueType val) { return abs(val); });
    } else {
        convert_to_bccoo_blk_blk(exec, source, result,
                                 [](ValueType val) { return abs(val); });
    }
}
/*
{
    size_type block_size_src = source->get_block_size();
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

    size_type block_size_res = source->get_block_size();
    auto* rows_data_res = result->get_rows();
    auto* offsets_data_res = result->get_offsets();
    auto* chunk_data_res = result->get_chunk();
    remove_complex<ValueType> val_res;

    if (num_stored_elements > 0) {
        offsets_data_res[0] = 0;
    }
    if (source->use_element_compression()) {
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
            get_detect_endblock(block_size_src, nblk_src, blk_src);
            put_detect_endblock(offsets_data_res, shf_res, block_size_res,
                                nblk_res, blk_res);
        }
    } else {
        auto* cols_data_src = source->get_const_cols();
        auto* types_data_src = source->get_const_types();

        auto* cols_data_res = result->get_cols();
        auto* types_data_res = result->get_types();

        blk_res = blk_src = 0;
        for (size_type i = 0; i < num_stored_elements; i += block_size_src) {
            size_type block_size_local =
                std::min(block_size_src, num_stored_elements - i);
            rows_data_res[blk_res] = rows_data_src[blk_src];
            cols_data_res[blk_res] = cols_data_src[blk_src];
            types_data_res[blk_res] = types_data_src[blk_src];
            auto type_blk = types_data_res[blk_res];
            if (type_blk & GKO_BCCOO_ROWS_MULTIPLE) {
                const uint8* rows_blk_src =
                    reinterpret_cast<const uint8*>(chunk_data_src) + shf_src;
                uint8* rows_blk_res =
                    reinterpret_cast<uint8*>(chunk_data_res) + shf_res;
                for (size_type j = 0; j < block_size_local; j++) {
                    rows_blk_res[j] = rows_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            }
            if (type_blk & GKO_BCCOO_COLS_8BITS) {
                const uint8* cols_blk_src =
                    reinterpret_cast<const uint8*>(chunk_data_src + shf_src);
                uint8* cols_blk_res =
                    reinterpret_cast<uint8*>(chunk_data_res + shf_res);
                for (size_type j = 0; j < block_size_local; j++) {
                    cols_blk_res[j] = cols_blk_src[j];
                }
                shf_src += block_size_local;
                shf_res += block_size_local;
            } else if (type_blk & GKO_BCCOO_COLS_16BITS) {
                std::memcpy(
                    reinterpret_cast<uint16*>(chunk_data_res + shf_res),
                    reinterpret_cast<const uint16*>(chunk_data_src + shf_src),
                    block_size_local * sizeof(uint16));
                shf_src += block_size_local * sizeof(uint16);
                shf_res += block_size_local * sizeof(uint16);
            } else {
                std::memcpy(
                    reinterpret_cast<uint32*>(chunk_data_res + shf_res),
                    reinterpret_cast<const uint32*>(chunk_data_src + shf_src),
                    block_size_local * sizeof(uint32));
                shf_src += block_size_local * sizeof(uint32);
                shf_res += block_size_local * sizeof(uint32);
            }
            if (true) {
                for (size_type j = 0; j < block_size_local; j++) {
                    val_src =
                        get_value_chunk<ValueType>(chunk_data_src, shf_src);
                    val_res = abs(val_src);
                    set_value_chunk<remove_complex<ValueType>>(
                        chunk_data_res, shf_res, val_res);
                    shf_src += sizeof(ValueType);
                    shf_res += sizeof(remove_complex<ValueType>);
                }
            }
            blk_src++;
            blk_res++;
            offsets_data_res[blk_res] = shf_res;
        }
    }
}
*/
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace reference
}  // namespace kernels
}  // namespace gko
