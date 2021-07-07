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

#include "core/matrix/coo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/format_conversion_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The Reference namespace.
 *
 * @ingroup reference
 */
namespace reference {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Coo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Coo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const ReferenceExecutor> exec,
           const matrix::Coo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    auto coo_val = a->get_const_values();
    auto coo_col = a->get_const_col_idxs();
    auto coo_row = a->get_const_row_idxs();
    auto num_cols = b->get_size()[1];
    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
        for (size_type j = 0; j < num_cols; j++) {
            c->at(coo_row[i], j) += coo_val[i] * b->at(coo_col[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Coo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    auto coo_val = a->get_const_values();
    auto coo_col = a->get_const_col_idxs();
    auto coo_row = a->get_const_row_idxs();
    auto alpha_val = alpha->at(0, 0);
    auto num_cols = b->get_size()[1];
    for (size_type i = 0; i < a->get_num_stored_elements(); i++) {
        for (size_type j = 0; j < num_cols; j++) {
            c->at(coo_row[i], j) +=
                alpha_val * coo_val[i] * b->at(coo_col[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Coo<ValueType, IndexType>* source,
                   matrix::Dense<ValueType>* result)
{
    auto coo_val = source->get_const_values();
    auto coo_col = source->get_const_col_idxs();
    auto coo_row = source->get_const_row_idxs();
    for (size_type i = 0; i < source->get_num_stored_elements(); i++) {
        result->at(coo_row[i], coo_col[i]) += coo_val[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_FILL_IN_DENSE_KERNEL);

/* */
template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Coo<ValueType, IndexType>* coo,
                    IndexType* rows, IndexType* offsets,
                    const size_type num_blocks, const size_type block_size,
                    size_type* mem_size)
{
    const IndexType* row_idxs = coo->get_const_row_idxs();
    const IndexType* col_idxs = coo->get_const_col_idxs();
    const size_type num_rows = coo->get_size()[0];
    const size_type num_stored_elements = coo->get_num_stored_elements();
    //    size_type num_blocks = rows.size();
    offsets[0] = 0;
    for (size_type b = 0; b < num_blocks; b++) {
        size_type p = 0;
        size_type k = b * block_size;
        size_type r = row_idxs[k];
        size_type c = 0;
        rows[b] = r;
        for (size_type l = 0; l < block_size && k < num_stored_elements;
             l++, k++) {
            if (row_idxs[k] != r) {  // new row
                r = row_idxs[k];
                c = 0;
                p++;
            }
            size_type d = col_idxs[k] - c;
            // if (d < 0x7D) { // When LUT is used
            if (d < 0xFD) {
                p++;
            } else if (d < 0xFFFF) {
                p += 3;
            } else {
                p += 5;
            }
            c = col_idxs[k];
            p += sizeof(ValueType);
        }
        offsets[b + 1] = p;
    }
    for (int b = 0; b < num_blocks; b++) offsets[b + 1] += offsets[b];
}
/* */
/*
template <typename ValueType, typename IndexType>
// template <typename IndexType>
void mem_size_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                    const IndexType *row_idxs, const IndexType *col_idxs,
                    const size_type num_rows, IndexType *rows,
                    IndexType *offsets, const size_type num_stored_elements,
                    const size_type num_blocks, const size_type block_size,
                    size_type *mem_size)  // GKO_NOT_IMPLEMENTED;

{
    //    size_type num_stored_elements = row_idxs.size();
    //    size_type num_blocks = rows.size();
    offsets[0] = 0;
    for (size_type b = 0; b < num_blocks; b++) {
        size_type p = 0;
        size_type k = b * block_size;
        size_type r = row_idxs[k];
        size_type c = 0;
        rows[b] = r;
        for (size_type l = 0; l < block_size && k < num_stored_elements;
             l++, k++) {
            if (row_idxs[k] != r) {  // new row
                r = row_idxs[k];
                c = 0;
                p++;
            }
            size_type d = col_idxs[k] - c;
            // if (d < 0x7D) { // When LUT is used
            if (d < 0xFD) {
                p++;
            } else if (d < 0xFFFF) {
                p += 3;
            } else {
                p += 5;
            }
            c = col_idxs[k];
        }
        offsets[b + 1] = p;
    }
    for (int b = 0; b < num_blocks; b++) offsets[b + 1] += offsets[b];
}
*/


// GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_MEM_SIZE_BCCOO_KERNEL);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void fill_bccoo(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Coo<ValueType, IndexType>* coo,
    //                const IndexType *row_idxs, const IndexType *col_idxs,
    //                const ValueType *values, const size_type num_rows,
    const IndexType* rows, const IndexType* offsets, uint8* data,
    //                const size_type num_stored_elements,
    const size_type num_blocks, const size_type block_size)
//  GKO_NOT_IMPLEMENTED;
/*  */
{
    const IndexType* row_idxs = coo->get_const_row_idxs();
    const IndexType* col_idxs = coo->get_const_col_idxs();
    const ValueType* values = coo->get_const_values();
    const size_type num_rows = coo->get_size()[0];
    const size_type num_stored_elements = coo->get_num_stored_elements();
    //    size_type num_blocks = rows.size();
    for (size_type b = 0; b < num_blocks; b++) {
        size_type p = offsets[b];
        size_type k = b * block_size;
        size_type r = row_idxs[k];
        size_type c = 0;
        for (size_type l = 0; l < block_size && k < num_stored_elements;
             l++, k++) {
            if (row_idxs[k] != r) {  // new row
                r = row_idxs[k];
                c = 0;
                data[p] = 0xFF;
                p++;
            }
            size_type d = col_idxs[k] - c;
            if (d < 0x7D) {
                data[p] = d;
                p++;
            } else if (d < 0xffff) {
                data[p] = 0xFD;
                p++;
                *(uint16*)(data + p) = d;
                p += 2;
            } else {
                data[p] = 0xFE;
                p++;
                *(uint32*)(data + p) = d;
                p += 4;
            }
            c = col_idxs[k];
            *(ValueType*)(data + p) = values[k];
            p += sizeof(ValueType);
        }
    }
}
/* */


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Coo<ValueType, IndexType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
// GKO_NOT_IMPLEMENTED;
/* */
{
    auto num_rows = result->get_size()[0];
    const auto nnz = result->get_num_stored_elements();
    const auto num_blocks = result->get_num_blocks();

    const auto source_row_idxs = source->get_const_row_idxs();
    const auto source_col_idxs = source->get_const_col_idxs();
    const auto source_values = source->get_const_values();

    const auto result_block_size = result->get_block_size();
    const auto result_rows = result->get_const_rows();
    const auto result_offsets = result->get_const_offsets();

    auto result_data = result->get_chunk();

    fill_bccoo(exec, source, result_rows, result_offsets, result_data,
               num_blocks, result_block_size);
    //    fill_bccoo(exec, source_row_idxs, source_col_idxs, source_values,
    //    num_rows,
    //               result_rows, result_offsets, result_data, nnz, num_blocks,
    //               result_block_size);
}
/* */

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Coo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto row_idxs = orig->get_const_row_idxs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const auto diag_size = diag->get_size()[0];
    const auto nnz = orig->get_num_stored_elements();
    auto diag_values = diag->get_values();

    for (size_type idx = 0; idx < nnz; idx++) {
        if (row_idxs[idx] == col_idxs[idx]) {
            diag_values[row_idxs[idx]] = values[idx];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_EXTRACT_DIAGONAL_KERNEL);


}  // namespace coo
}  // namespace reference
}  // namespace kernels
}  // namespace gko
