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


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"
#include "omp/components/atomic.hpp"


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


void get_default_block_size(std::shared_ptr<const OmpExecutor> exec,
                            size_type* block_size)
{
    *block_size = 10;
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
#pragma omp parallel for
    for (size_type i = 0; i < c->get_num_stored_elements(); i++) {
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
    auto beta_val = beta->at(0, 0);
#pragma omp parallel for
    for (size_type i = 0; i < c->get_num_stored_elements(); i++) {
        c->at(i) *= beta_val;
    }

    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);

#define OPTION 1

template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const OmpExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto num_blks = a->get_num_blocks();

// Computation of chunk
// #pragma omp parallel default(none), shared(exec, a, b, c, num_blks)
#pragma omp parallel default(shared)
    {
        auto num_cols = b->get_size()[1];
        array<ValueType> sumV_array(exec, num_cols);
#pragma omp for
        for (size_type blk = 0; blk < num_blks; blk++) {
            auto* rows_data = a->get_const_rows();
            auto* offsets_data = a->get_const_offsets();
            auto* chunk_data = a->get_const_chunk();
            auto block_size = a->get_block_size();
            size_type nblk = 0, col = 0;
            size_type row = rows_data[blk], row_old = 0;
            size_type shf = offsets_data[blk];
            ValueType val;
            ValueType* sumV = sumV_array.get_data();
            for (size_type j = 0; j < num_cols; j++) {
                sumV[j] = zero<ValueType>();
            }
            while (shf < offsets_data[blk + 1]) {
                row_old = row;
                uint8 ind = get_position_newrow(chunk_data, shf, row, col);
                get_next_position_value(chunk_data, nblk, ind, shf, col, val);
                if (row_old != row) {
#ifdef OPTION
                    for (size_type j = 0; j < num_cols; j++) {
                        atomic_add(c->at(row_old, j), sumV[j]);
                        // c->at(row_old, j) += sumV[j];
                        sumV[j] = zero<ValueType>();
                    }
#else
#pragma omp critical(bccoo_apply)
                    {
                        for (size_type j = 0; j < num_cols; j++) {
                            // TODO: replace with
                            // 	 OMP atomic_add (ValueType &inout,
                            //                   ValueType out);
                            //   atomic_add(c->at(row_old, j), sumV[j]);
                            c->at(row_old, j) += sumV[j];
                            sumV[j] = zero<ValueType>();
                        }
                    }
#endif
                }
                for (size_type j = 0; j < num_cols; j++) {
                    sumV[j] += val * b->at(col, j);
                }
            }
#ifdef OPTION
            for (size_type j = 0; j < num_cols; j++) {
                atomic_add(c->at(row, j), sumV[j]);
                // c->at(row_old, j) += sumV[j];
            }
#else
#pragma omp critical(bccoo_apply)
            {
                for (size_type j = 0; j < num_cols; j++) {
                    // TODO: replace with
                    // 	 OMP atomic_add (ValueType &inout, ValueType out);
                    //   atomic_add(c->at(row_old, j), sumV[j]);
                    c->at(row, j) += sumV[j];
                }
            }
#endif
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
    auto num_blks = a->get_num_blocks();

// Computation of chunk
// #pragma omp parallel default(none), shared(exec, alpha, a, b, c, num_blks)
#pragma omp parallel default(shared)
    {
        auto num_cols = b->get_size()[1];
        array<ValueType> sumV_array(exec, num_cols);
#pragma omp for
        for (size_type blk = 0; blk < num_blks; blk++) {
            auto* rows_data = a->get_const_rows();
            auto* offsets_data = a->get_const_offsets();
            auto* chunk_data = a->get_const_chunk();
            auto num_cols = b->get_size()[1];
            auto block_size = a->get_block_size();
            auto alpha_val = alpha->at(0, 0);
            size_type nblk = 0, col = 0;
            size_type row = rows_data[blk], row_old = 0;
            size_type shf = offsets_data[blk];
            ValueType val;
            ValueType* sumV = sumV_array.get_data();
            for (size_type j = 0; j < num_cols; j++) {
                sumV[j] = zero<ValueType>();
            }
            while (shf < offsets_data[blk + 1]) {
                row_old = row;
                uint8 ind = get_position_newrow(chunk_data, shf, row, col);
                get_next_position_value(chunk_data, nblk, ind, shf, col, val);
                if (row_old != row) {
#ifdef OPTION
                    for (size_type j = 0; j < num_cols; j++) {
                        atomic_add(c->at(row_old, j), sumV[j]);
                        // c->at(row_old, j) += sumV[j];
                        sumV[j] = zero<ValueType>();
                    }
#else
#pragma omp critical(bccoo_apply)
                    {
                        for (size_type j = 0; j < num_cols; j++) {
                            // TODO: replace with
                            // 	 OMP atomic_add (ValueType &inout,
                            //                   ValueType out);
                            //   atomic_add(c->at(row_old, j), sumV[j]);
                            c->at(row_old, j) += sumV[j];
                            sumV[j] = zero<ValueType>();
                        }
                    }
#endif
                }
                for (size_type j = 0; j < num_cols; j++) {
                    sumV[j] += alpha_val * val * b->at(col, j);
                }
            }
#ifdef OPTION
            for (size_type j = 0; j < num_cols; j++) {
                atomic_add(c->at(row, j), sumV[j]);
                // c->at(row_old, j) += sumV[j];
            }
#else
#pragma omp critical(bccoo_apply)
            {
                for (size_type j = 0; j < num_cols; j++) {
                    // TODO: replace with
                    // 	 OMP atomic_add (ValueType &inout, ValueType out);
                    //   atomic_add(c->at(row_old, j), sumV[j]);
                    c->at(row, j) += sumV[j];
                }
            }
#endif
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_compression(std::shared_ptr<const OmpExecutor> exec,
                            const matrix::Bccoo<ValueType, IndexType>* source,
                            matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COMPRESSION_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
{
    auto num_blk_src = source->get_num_blocks();
    if (source->get_block_size() == source->get_block_size()) {
        if (source->get_num_stored_elements() > 0) {
            result->get_offsets()[0] = 0;
        }
// #pragma omp parallel for default(none), shared(source, result, num_blk_src)
#pragma omp parallel for default(shared)
        for (size_type blk_src = 0; blk_src < num_blk_src; blk_src++) {
            auto* rows_data_src = source->get_const_rows();
            auto* offsets_data_src = source->get_const_offsets();
            auto* rows_data_res = result->get_rows();
            auto* offsets_data_res = result->get_offsets();
            rows_data_res[blk_src] = rows_data_src[blk_src];
            offsets_data_res[blk_src + 1] = offsets_data_src[blk_src + 1];
        }
// Computation of chunk
// #pragma omp parallel for default(none), shared(source, result, num_blk_src)
#pragma omp parallel for default(shared)
        for (size_type blk_src = 0; blk_src < num_blk_src; blk_src++) {
            auto* rows_data_src = source->get_const_rows();
            auto* offsets_data_src = source->get_const_offsets();
            auto* chunk_data_src = source->get_const_chunk();
            size_type block_sizeS = source->get_block_size();
            size_type num_bytes_src = source->get_num_bytes();
            size_type nblk_src = 0, col_src = 0;
            size_type row_src = rows_data_src[blk_src];
            size_type shf_src = offsets_data_src[blk_src];
            ValueType val_src;

            auto* rows_data_res = result->get_rows();
            auto* offsets_data_res = result->get_offsets();
            auto* chunk_data_res = result->get_chunk();
            size_type block_sizeR = result->get_block_size();
            size_type num_bytes_res = result->get_num_bytes();
            size_type nblk_res = 0, col_res = 0;
            size_type blk_res = blk_src;
            size_type row_res = row_src;
            size_type shf_res = shf_src;
            next_precision<ValueType> val_res;

            while (shf_src < offsets_data_src[blk_src + 1]) {
                uint8 ind_src = get_position_newrow_put(
                    chunk_data_src, shf_src, row_src, col_src, chunk_data_res,
                    nblk_res, blk_res, rows_data_res, shf_res, row_res,
                    col_res);
                get_next_position_value(chunk_data_src, nblk_src, ind_src,
                                        shf_src, col_src, val_src);
                val_res = (val_src);
                put_next_position_value(chunk_data_res, nblk_res,
                                        col_src - col_res, shf_res, col_res,
                                        val_res);
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
    auto num_blks = source->get_num_blocks();

// #pragma omp parallel for default(none), shared(source, result, num_blks)
#pragma omp parallel for default(shared)
    for (size_type blk = 0; blk < num_blks; blk++) {
        auto* rows_data = source->get_const_rows();
        auto* offsets_data = source->get_const_offsets();
        auto* chunk_data = source->get_const_chunk();
        auto row_idxs = result->get_row_idxs();
        auto col_idxs = result->get_col_idxs();
        auto values = result->get_values();
        size_type block_size = source->get_block_size();
        size_type nblk = 0, col = 0;
        size_type row = rows_data[blk];
        size_type shf = offsets_data[blk];
        size_type i = block_size * blk;
        ValueType val;
        while (shf < offsets_data[blk + 1]) {
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            row_idxs[i] = row;
            col_idxs[i] = col;
            values[i] = val;
            i++;
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
    const auto nnz = source->get_num_stored_elements();
    const auto num_blks = source->get_num_blocks();
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];

    array<IndexType> rows_array(exec, nnz);
    IndexType* row_idxs = rows_array.get_data();

// #pragma omp parallel for default(none), \
//     shared(source, result, row_idxs, num_blks)
#pragma omp parallel for default(shared)
    for (size_type blk = 0; blk < num_blks; blk++) {
        auto* rows_data = source->get_const_rows();
        auto* offsets_data = source->get_const_offsets();
        auto* chunk_data = source->get_const_chunk();
        auto col_idxs = result->get_col_idxs();
        auto values = result->get_values();
        size_type block_size = source->get_block_size();
        size_type nblk = 0, col = 0;
        size_type row = rows_data[blk];
        size_type shf = offsets_data[blk];
        size_type i = block_size * blk;
        ValueType val;
        while (shf < offsets_data[blk + 1]) {
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            row_idxs[i] = row;
            col_idxs[i] = col;
            values[i] = val;
            i++;
        }
    }
    auto row_ptrs = result->get_row_ptrs();
    components::convert_idxs_to_ptrs(exec, row_idxs, nnz, num_rows + 1,
                                     row_ptrs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_blks = source->get_num_blocks();

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type col = 0; col < num_cols; col++) {
            result->at(row, col) = zero<ValueType>();
        }
    }

// #pragma omp parallel for default(none), shared(source, result, num_blks)
#pragma omp parallel for default(shared)
    for (size_type blk = 0; blk < num_blks; blk++) {
        auto* rows_data = source->get_const_rows();
        auto* offsets_data = source->get_const_offsets();
        auto* chunk_data = source->get_const_chunk();
        size_type block_size = source->get_block_size();
        size_type nblk = 0, col = 0;
        size_type row = rows_data[blk];
        size_type shf = offsets_data[blk];
        ValueType val;
        while (shf < offsets_data[blk + 1]) {
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            result->at(row, col) += val;
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
    auto diag_values = diag->get_values();
    auto num_rows = diag->get_size()[0];
    auto num_blks = orig->get_num_blocks();

    for (size_type row = 0; row < num_rows; row++) {
        diag_values[row] = zero<ValueType>();
    }

// #pragma omp parallel for default(none), shared(diag_values, orig, num_blks)
#pragma omp parallel for default(shared)
    for (size_type blk = 0; blk < num_blks; blk++) {
        auto* rows_data = orig->get_const_rows();
        auto* offsets_data = orig->get_const_offsets();
        auto* chunk_data = orig->get_const_chunk();
        size_type block_size = orig->get_block_size();
        size_type nblk = 0, col = 0;
        size_type row = rows_data[blk];
        size_type shf = offsets_data[blk];
        ValueType val;
        while (shf < offsets_data[blk + 1]) {
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value(chunk_data, nblk, ind, shf, col, val);
            if (row == col) {
                diag_values[row] = val;
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
    auto num_blks = matrix->get_num_blocks();

// Computation of chunk
// #pragma omp parallel for default(none), shared(matrix, num_blks)
#pragma omp parallel for default(shared)
    for (size_type blk = 0; blk < num_blks; blk++) {
        auto* rows_data = matrix->get_const_rows();
        auto* offsets_data = matrix->get_const_offsets();
        auto* chunk_data = matrix->get_chunk();
        size_type block_size = matrix->get_block_size();
        size_type nblk = 0, col = 0;
        size_type row = rows_data[blk];
        size_type shf = offsets_data[blk];
        ValueType val;
        while (shf < offsets_data[blk + 1]) {
            uint8 ind = get_position_newrow(chunk_data, shf, row, col);
            get_next_position_value_put(chunk_data, nblk, ind, shf, col, val,
                                        [](ValueType val) { return abs(val); });
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
    auto num_blk_src = source->get_num_blocks();
    if (source->get_block_size() == source->get_block_size()) {
        if (source->get_num_stored_elements() > 0) {
            result->get_offsets()[0] = 0;
        }
// #pragma omp parallel for default(none), shared(source, result, num_blk_src)
#pragma omp parallel for default(shared)
        for (size_type blk_src = 0; blk_src < num_blk_src; blk_src++) {
            auto* rows_data_src = source->get_const_rows();
            auto* offsets_data_src = source->get_const_offsets();
            auto* rows_data_res = result->get_rows();
            auto* offsets_data_res = result->get_offsets();
            rows_data_res[blk_src] = rows_data_src[blk_src];
            offsets_data_res[blk_src + 1] = offsets_data_src[blk_src + 1];
        }
// Computation of chunk
// #pragma omp parallel for default(none), shared(source, result, num_blk_src)
#pragma omp parallel for default(shared)
        for (size_type blk_src = 0; blk_src < num_blk_src; blk_src++) {
            auto* rows_data_src = source->get_const_rows();
            auto* offsets_data_src = source->get_const_offsets();
            auto* chunk_data_src = source->get_const_chunk();
            size_type block_sizeS = source->get_block_size();
            size_type num_bytes_src = source->get_num_bytes();
            size_type nblk_src = 0, col_src = 0;
            size_type row_src = rows_data_src[blk_src];
            size_type shf_src = offsets_data_src[blk_src];
            ValueType val_src;

            auto* rows_data_res = result->get_rows();
            auto* offsets_data_res = result->get_offsets();
            auto* chunk_data_res = result->get_chunk();
            size_type block_sizeR = result->get_block_size();
            size_type num_bytes_res = result->get_num_bytes();
            size_type nblk_res = 0, col_res = 0;
            size_type blk_res = blk_src;
            size_type row_res = row_src;
            size_type shf_res = shf_src;
            remove_complex<ValueType> val_res;

            while (shf_src < offsets_data_src[blk_src + 1]) {
                uint8 ind_src = get_position_newrow_put(
                    chunk_data_src, shf_src, row_src, col_src, chunk_data_res,
                    nblk_res, blk_res, rows_data_res, shf_res, row_res,
                    col_res);
                get_next_position_value(chunk_data_src, nblk_src, ind_src,
                                        shf_src, col_src, val_src);
                val_res = abs(val_src);
                put_next_position_value(chunk_data_res, nblk_res,
                                        col_src - col_res, shf_res, col_res,
                                        val_res);
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
