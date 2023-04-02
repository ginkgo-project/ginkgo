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

#include "core/matrix/bccoo_kernels.hpp"


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "core/matrix/bccoo_memsize_convert.hpp"
#include "omp/components/atomic.hpp"
#include "omp/matrix/bccoo_helper.hpp"


namespace gko {
namespace kernels {
/**
 * @brief OpenMP namespace.
 *
 * @ingroup omp
 */
namespace omp {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


using namespace matrix::bccoo;


template <typename IndexType>
void get_default_block_size(std::shared_ptr<const OmpExecutor> exec,
                            IndexType* block_size)
{
    *block_size = 32;
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_GET_DEFAULT_BLOCK_SIZE_KERNEL);


void get_default_compression(std::shared_ptr<const OmpExecutor> exec,
                             compression* compression)
{
    *compression = compression::element;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
#pragma omp parallel for
    for (IndexType i = 0; i < c->get_num_stored_elements(); i++) {
        c->at(i) = zero<ValueType>();
    }

    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    ValueType beta_val = beta->at(0, 0);
#pragma omp parallel for
    for (IndexType i = 0; i < c->get_num_stored_elements(); i++) {
        c->at(i) *= beta_val;
    }

    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    IndexType num_blks = a->get_num_blocks();
    if (a->use_element_compression()) {
        // For element compression objects
#pragma omp parallel default(shared)
        {
            IndexType num_cols = b->get_size()[1];
            array<ValueType> sumV_array(exec, num_cols);
            // Each block is computed separately
#pragma omp for
            for (IndexType blk = 0; blk < num_blks; blk++) {
                const IndexType* rows_data = a->get_const_rows();
                const size_type* offsets_data = a->get_const_offsets();
                const uint8* chunk_data = a->get_const_chunk();
                const IndexType block_size = a->get_block_size();
                bool new_elm = false;
                IndexType row_old = 0;
                ValueType val;

                ValueType* sumV = sumV_array.get_data();
                // The auxiliary vector is initialized to zero
                for (IndexType j = 0; j < num_cols; j++) {
                    sumV[j] = zero<ValueType>();
                }

                compr_idxs<IndexType> idxs(blk, offsets_data[blk],
                                           rows_data[blk]);
                while (idxs.shf < offsets_data[blk + 1]) {
                    row_old = idxs.row;
                    uint8 ind = get_position_newrow(chunk_data, idxs);
                    get_next_position_value(chunk_data, ind, idxs, val);
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
                }
                if (new_elm) {
                    // If some values are processed and not accumulated,
                    // the computed values have to be accumulated to c
                    for (IndexType j = 0; j < num_cols; j++) {
                        atomic_add(c->at(idxs.row, j), sumV[j]);
                    }
                }
            }
        }
    } else {
        // For block compression objects
#pragma omp parallel default(shared)
        {
            IndexType num_cols = b->get_size()[1];
            array<ValueType> sumV_array(exec, num_cols);
            // Each block is computed separately
#pragma omp for
            for (IndexType blk = 0; blk < num_blks; blk++) {
                const IndexType* rows_data = a->get_const_rows();
                const IndexType* cols_data = a->get_const_cols();
                const uint8* types_data = a->get_const_types();
                const size_type* offsets_data = a->get_const_offsets();
                const uint8* chunk_data = a->get_const_chunk();

                const IndexType block_size = a->get_block_size();
                const IndexType num_stored_elements =
                    a->get_num_stored_elements();

                ValueType* sumV = sumV_array.get_data();
                // The auxiliary vector is initialized to zero
                for (IndexType j = 0; j < num_cols; j++) {
                    sumV[j] = zero<ValueType>();
                }

                const IndexType block_size_local = std::min(
                    block_size, num_stored_elements - blk * block_size);
                const uint8 type_blk = types_data[blk];
                compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
                compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                                   block_size_local, idxs,
                                                   types_data[idxs.blk]);
                if (blk_idxs.is_multi_row()) {
                    if (blk_idxs.is_row_16bits()) {
                        if (blk_idxs.is_column_8bits()) {
                            loop_block_multi_row<uint16, uint8, IndexType>(
                                chunk_data, block_size_local, b, c, idxs,
                                blk_idxs, sumV);
                        } else if (blk_idxs.is_column_16bits()) {
                            loop_block_multi_row<uint16, uint16, IndexType>(
                                chunk_data, block_size_local, b, c, idxs,
                                blk_idxs, sumV);
                        } else {
                            loop_block_multi_row<uint16, uint32, IndexType>(
                                chunk_data, block_size_local, b, c, idxs,
                                blk_idxs, sumV);
                        }
                    } else {
                        if (blk_idxs.is_column_8bits()) {
                            loop_block_multi_row<uint8, uint8, IndexType>(
                                chunk_data, block_size_local, b, c, idxs,
                                blk_idxs, sumV);
                        } else if (blk_idxs.is_column_16bits()) {
                            loop_block_multi_row<uint8, uint16, IndexType>(
                                chunk_data, block_size_local, b, c, idxs,
                                blk_idxs, sumV);
                        } else {
                            loop_block_multi_row<uint8, uint32, IndexType>(
                                chunk_data, block_size_local, b, c, idxs,
                                blk_idxs, sumV);
                        }
                    }
                } else {
                    if (blk_idxs.is_column_8bits()) {
                        loop_block_single_row<uint8, IndexType>(
                            chunk_data, block_size_local, b, c, idxs, blk_idxs,
                            sumV);
                    } else if (blk_idxs.is_column_16bits()) {
                        loop_block_single_row<uint16, IndexType>(
                            chunk_data, block_size_local, b, c, idxs, blk_idxs,
                            sumV);
                    } else {
                        loop_block_single_row<uint32, IndexType>(
                            chunk_data, block_size_local, b, c, idxs, blk_idxs,
                            sumV);
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    const IndexType num_blks = a->get_num_blocks();

    if (a->use_element_compression()) {
        // For element compression objects
#pragma omp parallel default(shared)
        {
            const IndexType num_cols = b->get_size()[1];
            array<ValueType> sumV_array(exec, num_cols);
            // Each block is computed separately
#pragma omp for
            for (IndexType blk = 0; blk < num_blks; blk++) {
                const IndexType* rows_data = a->get_const_rows();
                const size_type* offsets_data = a->get_const_offsets();
                const uint8* chunk_data = a->get_const_chunk();
                const IndexType num_cols = b->get_size()[1];
                const IndexType block_size = a->get_block_size();
                bool new_elm = false;
                const ValueType alpha_val = alpha->at(0, 0);
                IndexType row_old = 0;
                ValueType val;

                ValueType* sumV = sumV_array.get_data();
                // The auxiliary vector is initialized to zero
                for (IndexType j = 0; j < num_cols; j++) {
                    sumV[j] = zero<ValueType>();
                }

                compr_idxs<IndexType> idxs(blk, offsets_data[blk],
                                           rows_data[blk]);
                while (idxs.shf < offsets_data[blk + 1]) {
                    row_old = idxs.row;
                    uint8 ind = get_position_newrow(chunk_data, idxs);
                    get_next_position_value(chunk_data, ind, idxs, val);
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
                        sumV[j] += alpha_val * val * b->at(idxs.col, j);
                    }
                    new_elm = true;
                }
                if (new_elm) {
                    // If some values are processed and not accumulated,
                    // the computed values have to be accumulated to c
                    for (IndexType j = 0; j < num_cols; j++) {
                        atomic_add(c->at(idxs.row, j), sumV[j]);
                    }
                }
            }
        }
    } else {
#pragma omp parallel default(shared)
        {
            const IndexType num_cols = b->get_size()[1];
            array<ValueType> sumV_array(exec, num_cols);
            // Each block is computed separately
#pragma omp for
            for (IndexType blk = 0; blk < num_blks; blk++) {
                const IndexType* rows_data = a->get_const_rows();
                const IndexType* cols_data = a->get_const_cols();
                const uint8* types_data = a->get_const_types();
                const size_type* offsets_data = a->get_const_offsets();
                const uint8* chunk_data = a->get_const_chunk();

                const IndexType block_size = a->get_block_size();
                const IndexType num_stored_elements =
                    a->get_num_stored_elements();
                const ValueType alpha_val = alpha->at(0, 0);

                ValueType* sumV = sumV_array.get_data();
                // The auxiliary vector is initialized to zero
                for (IndexType j = 0; j < num_cols; j++) {
                    sumV[j] = zero<ValueType>();
                }

                const IndexType block_size_local = std::min(
                    block_size, num_stored_elements - blk * block_size);
                const uint8 type_blk = types_data[blk];
                compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
                compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                                   block_size_local, idxs,
                                                   types_data[idxs.blk]);
                if (blk_idxs.is_multi_row()) {
                    if (blk_idxs.is_row_16bits()) {
                        if (blk_idxs.is_column_8bits()) {
                            loop_block_multi_row<uint16, uint8, IndexType>(
                                chunk_data, block_size_local, alpha_val, b, c,
                                idxs, blk_idxs, sumV);
                        } else if (blk_idxs.is_column_16bits()) {
                            loop_block_multi_row<uint16, uint16, IndexType>(
                                chunk_data, block_size_local, alpha_val, b, c,
                                idxs, blk_idxs, sumV);
                        } else {
                            loop_block_multi_row<uint16, uint32, IndexType>(
                                chunk_data, block_size_local, alpha_val, b, c,
                                idxs, blk_idxs, sumV);
                        }
                    } else {
                        if (blk_idxs.is_column_8bits()) {
                            loop_block_multi_row<uint8, uint8, IndexType>(
                                chunk_data, block_size_local, alpha_val, b, c,
                                idxs, blk_idxs, sumV);
                        } else if (blk_idxs.is_column_16bits()) {
                            loop_block_multi_row<uint8, uint16, IndexType>(
                                chunk_data, block_size_local, alpha_val, b, c,
                                idxs, blk_idxs, sumV);
                        } else {
                            loop_block_multi_row<uint8, uint32, IndexType>(
                                chunk_data, block_size_local, alpha_val, b, c,
                                idxs, blk_idxs, sumV);
                        }
                    }
                } else {
                    if (blk_idxs.is_column_8bits()) {
                        loop_block_single_row<uint8, IndexType>(
                            chunk_data, block_size_local, alpha_val, b, c, idxs,
                            blk_idxs, sumV);
                    } else if (blk_idxs.is_column_16bits()) {
                        loop_block_single_row<uint16, IndexType>(
                            chunk_data, block_size_local, alpha_val, b, c, idxs,
                            blk_idxs, sumV);
                    } else {
                        loop_block_single_row<uint32, IndexType>(
                            chunk_data, block_size_local, alpha_val, b, c, idxs,
                            blk_idxs, sumV);
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    compression compress_res, const IndexType block_size_res,
                    size_type* mem_size)
{
    // This code is exactly equal to the reference executor
    if ((source->get_block_size() == block_size_res) &&
        (source->get_compression() == compress_res)) {
        *mem_size = source->get_num_bytes();
    } else if ((source->use_element_compression()) &&
               (compress_res == source->get_compression())) {
        mem_size_bccoo_elm_elm(exec, source, block_size_res, mem_size);
    } else if (source->use_element_compression()) {
        mem_size_bccoo_elm_blk(exec, source, block_size_res, mem_size);
    } else if (compress_res == compression::element) {
        mem_size_bccoo_blk_elm(exec, source, block_size_res, mem_size);
    } else {
        mem_size_bccoo_blk_blk(exec, source, block_size_res, mem_size);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_BCCOO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
{
    const IndexType num_blk_src = source->get_num_blocks();
    const IndexType num_stored_elements_src = source->get_num_stored_elements();

    // If the block_size or the compression are different in source
    // and result, the code is exactly equal to the reference executor
    if ((source->get_block_size() != result->get_block_size()) ||
        (source->get_compression() != result->get_compression())) {
        auto compress_res = result->get_compression();
        if ((source->use_element_compression()) &&
            (result->use_element_compression())) {
            convert_to_bccoo_elm_elm(exec, source, result,
                                     [](ValueType val) { return val; });
        } else if (source->use_element_compression()) {
            convert_to_bccoo_elm_blk(exec, source, result);
        } else if (compress_res == compression::element) {
            convert_to_bccoo_blk_elm(exec, source, result);
        } else {
            convert_to_bccoo_blk_blk(exec, source, result,
                                     [](ValueType val) { return val; });
        }
    } else {
        if (source->get_num_stored_elements() > 0) {
            result->get_offsets()[0] = 0;
        }
        if (source->use_element_compression()) {
            // For element compression objects
            // First, the non chunk vectors are copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const size_type* offsets_data_src = source->get_const_offsets();
                IndexType* rows_data_res = result->get_rows();
                size_type* offsets_data_res = result->get_offsets();
                const IndexType block_size_src = source->get_block_size();
                const IndexType block_size_local =
                    std::min(block_size_src, num_stored_elements_src -
                                                 blk_src * block_size_src);
                rows_data_res[blk_src] = rows_data_src[blk_src];
                offsets_data_res[blk_src + 1] =
                    offsets_data_src[blk_src + 1] +
                    ((blk_src + 1) * block_size_src *
                     sizeof(next_precision<ValueType>)) -
                    ((blk_src + 1) * block_size_src * sizeof(ValueType));
            }
            // Finally, the chunk vector is copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const size_type* offsets_data_src = source->get_const_offsets();
                const uint8* chunk_data_src = source->get_const_chunk();
                const IndexType block_size_src = source->get_block_size();
                const size_type num_bytes_src = source->get_num_bytes();
                ValueType val_src;
                compr_idxs<IndexType> idxs_src(blk_src,
                                               offsets_data_src[blk_src],
                                               offsets_data_src[blk_src]);

                IndexType* rows_data_res = result->get_rows();
                size_type* offsets_data_res = result->get_offsets();
                uint8* chunk_data_res = result->get_chunk();
                const IndexType block_size_res = result->get_block_size();
                const size_type num_bytes_res = result->get_num_bytes();
                const IndexType blk_res = blk_src;
                next_precision<ValueType> val_res;
                compr_idxs<IndexType> idxs_res(blk_res,
                                               offsets_data_res[blk_res],
                                               offsets_data_res[blk_res]);
                while (idxs_src.shf < offsets_data_src[blk_src + 1]) {
                    uint8 ind_src = get_position_newrow_put(
                        chunk_data_src, idxs_src, chunk_data_res, rows_data_res,
                        idxs_res);
                    get_next_position_value(chunk_data_src, ind_src, idxs_src,
                                            val_src);
                    val_res = (val_src);
                    put_next_position_value(chunk_data_res,
                                            idxs_src.col - idxs_res.col,
                                            val_res, idxs_res);
                }
            }
        } else {
            // For block compression objects
            // First, the non chunk vectors are copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const IndexType* cols_data_src = source->get_const_cols();
                const uint8* types_data_src = source->get_const_types();
                const size_type* offsets_data_src = source->get_const_offsets();
                IndexType* rows_data_res = result->get_rows();
                IndexType* cols_data_res = result->get_cols();
                uint8* types_data_res = result->get_types();
                size_type* offsets_data_res = result->get_offsets();
                const IndexType block_size_src = source->get_block_size();
                const IndexType block_size_local =
                    std::min(block_size_src, num_stored_elements_src -
                                                 blk_src * block_size_src);
                rows_data_res[blk_src] = rows_data_src[blk_src];
                cols_data_res[blk_src] = cols_data_src[blk_src];
                types_data_res[blk_src] = types_data_src[blk_src];
                offsets_data_res[blk_src + 1] =
                    offsets_data_src[blk_src + 1] +
                    ((blk_src + 1) * block_size_src *
                     sizeof(next_precision<ValueType>)) -
                    ((blk_src + 1) * block_size_src * sizeof(ValueType));
            }
            // Finally, the chunk vector is copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const IndexType* cols_data_src = source->get_const_cols();
                const uint8* types_data_src = source->get_const_types();
                const size_type* offsets_data_src = source->get_const_offsets();
                const uint8* chunk_data_src = source->get_const_chunk();
                const IndexType block_size_src = source->get_block_size();
                const size_type num_bytes_src = source->get_num_bytes();
                const IndexType block_size_local_src =
                    std::min(block_size_src, num_stored_elements_src -
                                                 blk_src * block_size_src);
                ValueType val_src;
                compr_idxs<IndexType> idxs_src(blk_src,
                                               offsets_data_src[blk_src]);
                compr_blk_idxs<IndexType> blk_idxs_src(
                    rows_data_src, cols_data_src, block_size_local_src,
                    idxs_src, types_data_src[idxs_src.blk]);

                IndexType* rows_data_res = result->get_rows();
                IndexType* cols_data_res = result->get_cols();
                uint8* types_data_res = result->get_types();
                size_type* offsets_data_res = result->get_offsets();
                uint8* chunk_data_res = result->get_chunk();
                const IndexType block_size_res = result->get_block_size();
                const size_type num_bytes_res = result->get_num_bytes();
                const IndexType block_size_local_res = block_size_local_src;
                next_precision<ValueType> val_res;
                compr_idxs<IndexType> idxs_res(blk_src,
                                               offsets_data_res[blk_src]);
                compr_blk_idxs<IndexType> blk_idxs_res(
                    rows_data_res, cols_data_res, block_size_local_res,
                    idxs_res, types_data_res[idxs_res.blk]);

                write_chunk_blk<IndexType, ValueType,
                                next_precision<ValueType>>(
                    idxs_src, blk_idxs_src, block_size_local_src,
                    chunk_data_src, idxs_res, blk_idxs_res,
                    block_size_local_res, chunk_data_res,
                    [](ValueType val) { return (val); });
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
{
    IndexType num_blks = source->get_num_blocks();

    if (source->use_element_compression()) {
        // For element compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = source->get_const_rows();
            const size_type* offsets_data = source->get_const_offsets();
            const uint8* chunk_data = source->get_const_chunk();
            IndexType* row_idxs = result->get_row_idxs();
            IndexType* col_idxs = result->get_col_idxs();
            ValueType* values = result->get_values();
            const IndexType block_size = source->get_block_size();
            compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
            IndexType i = block_size * blk;
            ValueType val;
            while (idxs.shf < offsets_data[blk + 1]) {
                uint8 ind = get_position_newrow(chunk_data, idxs);
                get_next_position_value(chunk_data, ind, idxs, val);
                row_idxs[i] = idxs.row;
                col_idxs[i] = idxs.col;
                values[i] = val;
                i++;
            }
        }
    } else {
        // For block compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = source->get_const_rows();
            const IndexType* cols_data = source->get_const_cols();
            const uint8* types_data = source->get_const_types();
            const size_type* offsets_data = source->get_const_offsets();
            const uint8* chunk_data = source->get_const_chunk();

            IndexType* row_idxs = result->get_row_idxs();
            IndexType* col_idxs = result->get_col_idxs();
            ValueType* values = result->get_values();

            const IndexType block_size = source->get_block_size();
            const size_type num_bytes = source->get_num_bytes();
            const IndexType num_stored_elements =
                source->get_num_stored_elements();
            const IndexType block_size_local =
                std::min(block_size, num_stored_elements - blk * block_size);
            IndexType pos = block_size * blk;

            ValueType val;
            compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
            compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                               block_size_local, idxs,
                                               types_data[idxs.blk]);
            for (IndexType i = 0; i < block_size_local; i++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs, val);
                row_idxs[pos + i] = idxs.row;
                col_idxs[pos + i] = idxs.col;
                values[pos + i] = val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const IndexType nnz = source->get_num_stored_elements();
    const IndexType num_blks = source->get_num_blocks();
    const IndexType num_rows = source->get_size()[0];
    const IndexType num_cols = source->get_size()[1];

    array<IndexType> rows_array(exec, nnz);
    IndexType* row_idxs = rows_array.get_data();

    if (source->use_element_compression()) {
        // For element compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = source->get_const_rows();
            const size_type* offsets_data = source->get_const_offsets();
            const uint8* chunk_data = source->get_const_chunk();
            IndexType* col_idxs = result->get_col_idxs();
            ValueType* values = result->get_values();
            const IndexType block_size = source->get_block_size();
            compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
            IndexType i = block_size * blk;
            ValueType val;
            while (idxs.shf < offsets_data[blk + 1]) {
                uint8 ind = get_position_newrow(chunk_data, idxs);
                get_next_position_value(chunk_data, ind, idxs, val);
                row_idxs[i] = idxs.row;
                col_idxs[i] = idxs.col;
                values[i] = val;
                i++;
            }
        }
    } else {
        // For block compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = source->get_const_rows();
            const IndexType* cols_data = source->get_const_cols();
            const uint8* types_data = source->get_const_types();
            const size_type* offsets_data = source->get_const_offsets();
            const uint8* chunk_data = source->get_const_chunk();

            IndexType* col_idxs = result->get_col_idxs();
            ValueType* values = result->get_values();

            const IndexType block_size = source->get_block_size();
            const size_type num_bytes = source->get_num_bytes();
            const IndexType num_stored_elements =
                source->get_num_stored_elements();
            const IndexType block_size_local =
                std::min(block_size, num_stored_elements - blk * block_size);
            IndexType pos = block_size * blk;

            ValueType val;
            compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
            compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                               block_size_local, idxs,
                                               types_data[idxs.blk]);
            for (IndexType i = 0; i < block_size_local; i++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs, val);
                row_idxs[pos + i] = idxs.row;
                col_idxs[pos + i] = idxs.col;
                values[pos + i] = val;
            }
        }
    }
    IndexType* row_ptrs = result->get_row_ptrs();
    components::convert_idxs_to_ptrs(exec, row_idxs, nnz, num_rows, row_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    const IndexType num_rows = result->get_size()[0];
    const IndexType num_cols = result->get_size()[1];
    const IndexType num_blks = source->get_num_blocks();

    // First, result is initialized to zero
#pragma omp parallel for default(shared)
    for (IndexType row = 0; row < num_rows; row++) {
        for (IndexType col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
    }

    if (source->use_element_compression()) {
        // For element compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = source->get_const_rows();
            const size_type* offsets_data = source->get_const_offsets();
            const uint8* chunk_data = source->get_const_chunk();
            const IndexType block_size = source->get_block_size();
            compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
            ValueType val;
            while (idxs.shf < offsets_data[blk + 1]) {
                uint8 ind = get_position_newrow(chunk_data, idxs);
                get_next_position_value(chunk_data, ind, idxs, val);
                result->at(idxs.row, idxs.col) += val;
            }
        }
    } else {
        // For block compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = source->get_const_rows();
            const IndexType* cols_data = source->get_const_cols();
            const uint8* types_data = source->get_const_types();
            const size_type* offsets_data = source->get_const_offsets();
            const uint8* chunk_data = source->get_const_chunk();

            const IndexType block_size = source->get_block_size();
            const size_type num_bytes = source->get_num_bytes();
            const IndexType num_stored_elements =
                source->get_num_stored_elements();
            const IndexType block_size_local =
                std::min(block_size, num_stored_elements - blk * block_size);
            IndexType pos = block_size * blk;

            ValueType val;
            compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
            compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                               block_size_local, idxs,
                                               types_data[idxs.blk]);
            for (IndexType i = 0; i < block_size_local; i++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs, val);
                result->at(idxs.row, idxs.col) += val;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    ValueType* diag_values = diag->get_values();
    const IndexType num_rows = diag->get_size()[0];
    const IndexType num_blks = orig->get_num_blocks();

    // First, diag is initialized to zero
#pragma omp parallel for default(shared)
    for (IndexType row = 0; row < num_rows; row++) {
        diag_values[row] = zero<ValueType>();
    }

    if (orig->use_element_compression()) {
        // For element compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = orig->get_const_rows();
            const size_type* offsets_data = orig->get_const_offsets();
            const uint8* chunk_data = orig->get_const_chunk();
            const IndexType block_size = orig->get_block_size();
            compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
            ValueType val;
            while (idxs.shf < offsets_data[blk + 1]) {
                uint8 ind = get_position_newrow(chunk_data, idxs);
                get_next_position_value(chunk_data, ind, idxs, val);
                if (idxs.row == idxs.col) {
                    diag_values[idxs.row] = val;
                }
            }
        }
    } else {
        // For block compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = orig->get_const_rows();
            const IndexType* cols_data = orig->get_const_cols();
            const uint8* types_data = orig->get_const_types();
            const size_type* offsets_data = orig->get_const_offsets();
            const uint8* chunk_data = orig->get_const_chunk();

            const IndexType block_size = orig->get_block_size();
            const size_type num_bytes = orig->get_num_bytes();
            const IndexType num_stored_elements =
                orig->get_num_stored_elements();
            const IndexType block_size_local =
                std::min(block_size, num_stored_elements - blk * block_size);
            IndexType pos = block_size * blk;

            ValueType val;
            compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
            compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                               block_size_local, idxs,
                                               types_data[idxs.blk]);
            for (IndexType i = 0; i < block_size_local; i++) {
                get_block_position_value<IndexType, ValueType>(
                    chunk_data, blk_idxs, idxs, val);
                if (idxs.row == idxs.col) {
                    diag_values[idxs.row] = val;
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const OmpExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
{
    IndexType num_blks = matrix->get_num_blocks();

    if (matrix->use_element_compression()) {
        // For element compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = matrix->get_const_rows();
            const size_type* offsets_data = matrix->get_const_offsets();
            uint8* chunk_data = matrix->get_chunk();
            const IndexType block_size = matrix->get_block_size();
            compr_idxs<IndexType> idxs(blk, offsets_data[blk], rows_data[blk]);
            ValueType val;
            while (idxs.shf < offsets_data[blk + 1]) {
                uint8 ind = get_position_newrow(chunk_data, idxs);
                get_next_position_value_put(
                    chunk_data, ind, idxs, val,
                    [](ValueType val) { return abs(val); });
            }
        }
    } else {
        // For block compression objects
        // Each block is computed separately
#pragma omp parallel for default(shared)
        for (IndexType blk = 0; blk < num_blks; blk++) {
            const IndexType* rows_data = matrix->get_const_rows();
            const IndexType* cols_data = matrix->get_const_cols();
            const uint8* types_data = matrix->get_const_types();
            const size_type* offsets_data = matrix->get_const_offsets();
            uint8* chunk_data = matrix->get_chunk();

            const IndexType block_size = matrix->get_block_size();
            const size_type num_bytes = matrix->get_num_bytes();
            const IndexType num_stored_elements =
                matrix->get_num_stored_elements();
            const IndexType block_size_local =
                std::min(block_size, num_stored_elements - blk * block_size);
            IndexType pos = block_size * blk;

            compr_idxs<IndexType> idxs(blk, offsets_data[blk]);
            compr_blk_idxs<IndexType> blk_idxs(rows_data, cols_data,
                                               block_size_local, idxs,
                                               types_data[idxs.blk]);
            for (IndexType i = 0; i < block_size_local; i++) {
                if (true) {
                    ValueType val;
                    val = get_value_chunk<ValueType>(chunk_data,
                                                     blk_idxs.shf_val);
                    val = abs(val);
                    set_value_chunk<ValueType>(chunk_data, blk_idxs.shf_val,
                                               val);
                    blk_idxs.shf_val += sizeof(ValueType);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    remove_complex<matrix::Bccoo<ValueType, IndexType>>* result)
{
    const IndexType num_blk_src = source->get_num_blocks();
    // If the block_size or the compression are different in source
    // and result, the code is exactly equal to the reference executor
    if ((source->get_block_size() != result->get_block_size()) ||
        (source->get_compression() != result->get_compression())) {
        if (source->use_element_compression()) {
            convert_to_bccoo_elm_elm(exec, source, result,
                                     [](ValueType val) { return abs(val); });
        } else {
            convert_to_bccoo_blk_blk(exec, source, result,
                                     [](ValueType val) { return abs(val); });
        }
    } else {
        if (source->get_num_stored_elements() > 0) {
            result->get_offsets()[0] = 0;
        }
        if (source->use_element_compression()) {
            // For element compression objects
            // First, the non chunk vectors are copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const size_type* offsets_data_src = source->get_const_offsets();
                IndexType* rows_data_res = result->get_rows();
                size_type* offsets_data_res = result->get_offsets();
                rows_data_res[blk_src] = rows_data_src[blk_src];
                offsets_data_res[blk_src + 1] = offsets_data_src[blk_src + 1];
            }
            // Finally, the chunk vector is copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const size_type* offsets_data_src = source->get_const_offsets();
                const uint8* chunk_data_src = source->get_const_chunk();
                const IndexType block_size_src = source->get_block_size();
                const size_type num_bytes_src = source->get_num_bytes();
                compr_idxs<IndexType> idxs_src(
                    blk_src, offsets_data_src[blk_src], rows_data_src[blk_src]);
                ValueType val_src;

                IndexType* rows_data_res = result->get_rows();
                size_type* offsets_data_res = result->get_offsets();
                uint8* chunk_data_res = result->get_chunk();
                const IndexType block_size_res = result->get_block_size();
                const size_type num_bytes_res = result->get_num_bytes();
                compr_idxs<IndexType> idxs_res(
                    blk_src, offsets_data_res[blk_src], rows_data_res[blk_src]);
                remove_complex<ValueType> val_res;

                while (idxs_src.shf < offsets_data_src[blk_src + 1]) {
                    uint8 ind_src = get_position_newrow_put(
                        chunk_data_src, idxs_src, chunk_data_res, rows_data_res,
                        idxs_res);
                    get_next_position_value(chunk_data_src, ind_src, idxs_src,
                                            val_src);
                    val_res = abs(val_src);
                    put_next_position_value(chunk_data_res,
                                            idxs_src.col - idxs_res.col,
                                            val_res, idxs_res);
                }
            }
        } else {
            // For block compression objects
            // size_type num_stored_elements_src =
            IndexType num_stored_elements_src =
                source->get_num_stored_elements();
            // First, the non chunk vectors are copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const IndexType* cols_data_src = source->get_const_cols();
                const uint8* types_data_src = source->get_const_types();
                const size_type* offsets_data_src = source->get_const_offsets();
                IndexType* rows_data_res = result->get_rows();
                IndexType* cols_data_res = result->get_cols();
                uint8* types_data_res = result->get_types();
                size_type* offsets_data_res = result->get_offsets();
                const IndexType block_size_src = source->get_block_size();
                const IndexType block_size_local =
                    std::min(block_size_src, num_stored_elements_src -
                                                 blk_src * block_size_src);
                rows_data_res[blk_src] = rows_data_src[blk_src];
                cols_data_res[blk_src] = cols_data_src[blk_src];
                types_data_res[blk_src] = types_data_src[blk_src];
                offsets_data_res[blk_src + 1] =
                    offsets_data_src[blk_src + 1] +
                    (block_size_local * sizeof(remove_complex<ValueType>)) -
                    (block_size_local * sizeof(ValueType));
            }
            // Finally, the chunk vector is copied
            // Each block is computed separately
#pragma omp parallel for default(shared)
            for (IndexType blk_src = 0; blk_src < num_blk_src; blk_src++) {
                const IndexType* rows_data_src = source->get_const_rows();
                const IndexType* cols_data_src = source->get_const_cols();
                const uint8* types_data_src = source->get_const_types();
                const size_type* offsets_data_src = source->get_const_offsets();
                const uint8* chunk_data_src = source->get_const_chunk();
                IndexType block_size_src = source->get_block_size();
                const size_type num_bytes_src = source->get_num_bytes();
                const IndexType block_size_local_src =
                    std::min(block_size_src, num_stored_elements_src -
                                                 blk_src * block_size_src);
                ValueType val_src;
                compr_idxs<IndexType> idxs_src(blk_src,
                                               offsets_data_src[blk_src]);
                compr_blk_idxs<IndexType> blk_idxs_src(
                    rows_data_src, cols_data_src, block_size_local_src,
                    idxs_src, types_data_src[idxs_src.blk]);

                IndexType* rows_data_res = result->get_rows();
                IndexType* cols_data_res = result->get_cols();
                uint8* types_data_res = result->get_types();
                size_type* offsets_data_res = result->get_offsets();
                uint8* chunk_data_res = result->get_chunk();
                const IndexType block_size_res = result->get_block_size();
                const size_type num_bytes_res = result->get_num_bytes();
                const IndexType block_size_local_res = block_size_local_src;
                remove_complex<ValueType> val_res;
                compr_idxs<IndexType> idxs_res(blk_src,
                                               offsets_data_res[blk_src]);
                compr_blk_idxs<IndexType> blk_idxs_res(
                    rows_data_res, cols_data_res, block_size_local_res,
                    idxs_res, types_data_res[idxs_res.blk]);
                write_chunk_blk<IndexType, ValueType,
                                remove_complex<ValueType>>(
                    idxs_src, blk_idxs_src, block_size_local_src,
                    chunk_data_src, idxs_res, blk_idxs_res,
                    block_size_local_res, chunk_data_res,
                    [](ValueType val) { return abs(val); });
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace omp
}  // namespace kernels
}  // namespace gko
