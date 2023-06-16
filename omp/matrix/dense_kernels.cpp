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

#include "core/matrix/dense_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
#include <ginkgo/core/matrix/bccoo.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>
#include <ginkgo/core/matrix/ell.hpp>
#include <ginkgo/core/matrix/fbcsr.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>
#include <ginkgo/core/matrix/sellp.hpp>
#include <ginkgo/core/matrix/sparsity_csr.hpp>


#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/bccoo_helper.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup dense
 */
namespace dense {


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Dense<ValueType>* x,
                          const matrix::Dense<ValueType>* y,
                          matrix::Dense<ValueType>* result, array<char>& tmp)
{
    // OpenMP uses the unified kernel.
    compute_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               const matrix::Dense<ValueType>* x,
                               const matrix::Dense<ValueType>* y,
                               matrix::Dense<ValueType>* result,
                               array<char>& tmp)
{
    compute_conj_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                            const matrix::Dense<ValueType>* x,
                            matrix::Dense<remove_complex<ValueType>>* result,
                            array<char>& tmp)
{
    compute_norm2(exec, x, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void simple_apply(std::shared_ptr<const DefaultExecutor> exec,
                  const matrix::Dense<ValueType>* a,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* c)
{
#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type col = 0; col < c->get_size()[1]; ++col) {
            c->at(row, col) = zero<ValueType>();
        }
    }

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type inner = 0; inner < a->get_size()[1]; ++inner) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) += a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const DefaultExecutor> exec,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* a, const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* c)
{
    if (is_nonzero(beta->at(0, 0))) {
#pragma omp parallel for
        for (size_type row = 0; row < c->get_size()[0]; ++row) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) *= beta->at(0, 0);
            }
        }
    } else {
#pragma omp parallel for
        for (size_type row = 0; row < c->get_size()[0]; ++row) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) *= zero<ValueType>();
            }
        }
    }

#pragma omp parallel for
    for (size_type row = 0; row < c->get_size()[0]; ++row) {
        for (size_type inner = 0; inner < a->get_size()[1]; ++inner) {
            for (size_type col = 0; col < c->get_size()[1]; ++col) {
                c->at(row, col) +=
                    alpha->at(0, 0) * a->at(row, inner) * b->at(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType>
void mem_size_bccoo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    const size_type block_size,
                    const matrix::bccoo::compression compress,
                    size_type* result)
{
    // This is the same code as in reference executor
    if (compress == matrix::bccoo::compression::element) {
        // For element compression objects
        auto num_rows = source->get_size()[0];
        auto num_cols = source->get_size()[1];
        // TODO: Compute num_nonzeros and return
        matrix::bccoo::compr_idxs<size_type> idxs;
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                if (source->at(row, col) != zero<ValueType>()) {
                    // Counting bytes to write (row,col,val) on result
                    matrix::bccoo::cnt_detect_newblock(row - idxs.row, idxs);
                    size_type col_src_res =
                        matrix::bccoo::cnt_position_newrow_mat_data(row, col,
                                                                    idxs);
                    matrix::bccoo::cnt_next_position_value(
                        col_src_res, source->at(row, col), idxs);
                    matrix::bccoo::cnt_detect_endblock(block_size, idxs);
                }
            }
        }
        *result = idxs.shf;
    } else {
        // For block compression objects
        auto num_rows = source->get_size()[0];
        auto num_cols = source->get_size()[1];
        // TODO: Compute num_nonzeros and return
        matrix::bccoo::compr_idxs<size_type> idxs;
        matrix::bccoo::compr_blk_idxs<size_type> blk_idxs;
        for (size_type row = 0; row < num_rows; ++row) {
            for (size_type col = 0; col < num_cols; ++col) {
                if (source->at(row, col) != zero<ValueType>()) {
                    matrix::bccoo::proc_block_indices<size_type>(row, col, idxs,
                                                                 blk_idxs);
                    idxs.nblk++;
                    if (idxs.nblk == block_size) {
                        // Counting bytes to write block on result
                        matrix::bccoo::cnt_block_indices<size_type, ValueType>(
                            block_size, blk_idxs, idxs);
                        idxs.blk++;
                        idxs.nblk = 0;
                        blk_idxs = {};
                    }
                }
            }
        }
        if (idxs.nblk > 0) {
            // Counting bytes to write block on result
            matrix::bccoo::cnt_block_indices<size_type, ValueType>(
                idxs.nblk, blk_idxs, idxs);
            idxs.blk++;
            idxs.nblk = 0;
            blk_idxs = {};
        }
        *result = idxs.shf;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_MEM_SIZE_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_bccoo(std::shared_ptr<const OmpExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Bccoo<ValueType, IndexType>* result)
{
    // This is the same code as in reference executor
    if (result->use_element_compression()) {
        // For element compression objects
        IndexType block_size = result->get_block_size();
        IndexType* rows_data = result->get_rows();
        size_type* offsets_data = result->get_offsets();
        uint8* compressed_data = result->get_compressed_data();

        auto num_rows = source->get_size()[0];
        auto num_cols = source->get_size()[1];

        auto num_stored_elements = result->get_num_stored_elements();
        matrix::bccoo::compr_idxs<IndexType> idxs = {};

        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (IndexType row = 0; row < num_rows; ++row) {
            for (IndexType col = 0; col < num_cols; ++col) {
                if (source->at(row, col) != zero<ValueType>()) {
                    // Writing (row,col,val) to result
                    matrix::bccoo::put_detect_newblock(
                        compressed_data, rows_data, row - idxs.row, idxs);
                    IndexType col_src_res =
                        matrix::bccoo::put_position_newrow_mat_data(
                            row, col, compressed_data, idxs);
                    matrix::bccoo::put_next_position_value(
                        compressed_data, col - idxs.col, source->at(row, col),
                        idxs);
                    matrix::bccoo::put_detect_endblock(offsets_data, block_size,
                                                       idxs);
                }
            }
        }
        if (idxs.nblk > 0) {
            offsets_data[idxs.blk + 1] = idxs.shf;
        }
    } else {
        // For block compression objects
        auto num_rows = source->get_size()[0];
        auto num_cols = source->get_size()[1];

        auto* rows_data = result->get_rows();
        auto* cols_data = result->get_cols();
        auto* types_data = result->get_types();
        auto* offsets_data = result->get_offsets();
        auto* compressed_data = result->get_compressed_data();

        auto num_stored_elements = result->get_num_stored_elements();
        auto block_size = result->get_block_size();

        matrix::bccoo::compr_idxs<IndexType> idxs;
        matrix::bccoo::compr_blk_idxs<IndexType> blk_idxs;
        uint8 type_blk = {};
        ValueType val;

        array<IndexType> rows_blk(exec, block_size);
        array<IndexType> cols_blk(exec, block_size);
        array<ValueType> vals_blk(exec, block_size);

        if (num_stored_elements > 0) {
            offsets_data[0] = 0;
        }
        for (IndexType row = 0; row < num_rows; ++row) {
            for (IndexType col = 0; col < num_cols; ++col) {
                if (source->at(row, col) != zero<ValueType>()) {
                    // Analyzing the impact of (row,col,val) in the block
                    matrix::bccoo::proc_block_indices<IndexType>(row, col, idxs,
                                                                 blk_idxs);
                    rows_blk.get_data()[idxs.nblk] = row;
                    cols_blk.get_data()[idxs.nblk] = col;
                    vals_blk.get_data()[idxs.nblk] = source->at(row, col);
                    idxs.nblk++;
                    if (idxs.nblk == block_size) {
                        // Writing block on result
                        type_blk =
                            matrix::bccoo::write_compressed_data_blk_type(
                                idxs, blk_idxs, rows_blk, cols_blk, vals_blk,
                                compressed_data);
                        rows_data[idxs.blk] = blk_idxs.row_frst;
                        cols_data[idxs.blk] = blk_idxs.col_frst;
                        types_data[idxs.blk] = type_blk;
                        offsets_data[++idxs.blk] = idxs.shf;
                        idxs.nblk = 0;
                        blk_idxs = {};
                    }
                }
            }
        }
        if (idxs.nblk > 0) {
            // Writing block on result
            type_blk = matrix::bccoo::write_compressed_data_blk_type(
                idxs, blk_idxs, rows_blk, cols_blk, vals_blk, compressed_data);
            rows_data[idxs.blk] = blk_idxs.row_frst;
            cols_data[idxs.blk] = blk_idxs.col_frst;
            types_data[idxs.blk] = type_blk;
            offsets_data[++idxs.blk] = idxs.shf;
            idxs.nblk = 0;
            blk_idxs = {};
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_BCCOO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    const int64* row_ptrs,
                    matrix::Coo<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto idxs = row_ptrs[row];
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (is_nonzero(val)) {
                row_idxs[idxs] = row;
                col_idxs[idxs] = col;
                values[idxs] = val;
                ++idxs;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_COO_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto cur_ptr = row_ptrs[row];
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (is_nonzero(val)) {
                col_idxs[cur_ptr] = col;
                values[cur_ptr] = val;
                ++cur_ptr;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* source,
                    matrix::Ell<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto max_nnz_per_row = result->get_num_stored_elements_per_row();
#pragma omp parallel for
    for (size_type i = 0; i < max_nnz_per_row; i++) {
        for (size_type j = 0; j < result->get_stride(); j++) {
            result->val_at(j, i) = zero<ValueType>();
            result->col_at(j, i) = invalid_index<IndexType>();
        }
    }
#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        size_type col_idx = 0;
        for (size_type col = 0; col < num_cols; col++) {
            auto val = source->at(row, col);
            if (is_nonzero(val)) {
                result->val_at(row, col_idx) = val;
                result->col_at(row, col_idx) = col;
                col_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Fbcsr<ValueType, IndexType>* result)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto bs = result->get_block_size();
    const auto nzbs = result->get_num_stored_blocks();
    const auto num_block_rows = num_rows / bs;
    const auto num_block_cols = num_cols / bs;
    acc::range<acc::block_col_major<ValueType, 3>> blocks(
        std::array<acc::size_type, 3>{static_cast<acc::size_type>(nzbs),
                                      static_cast<acc::size_type>(bs),
                                      static_cast<acc::size_type>(bs)},
        result->get_values());
    auto col_idxs = result->get_col_idxs();
#pragma omp parallel for
    for (size_type brow = 0; brow < num_block_rows; ++brow) {
        auto block = result->get_const_row_ptrs()[brow];
        for (size_type bcol = 0; bcol < num_block_cols; ++bcol) {
            bool block_nz = false;
            for (int lrow = 0; lrow < bs; ++lrow) {
                for (int lcol = 0; lcol < bs; ++lcol) {
                    const auto row = lrow + bs * brow;
                    const auto col = lcol + bs * bcol;
                    block_nz = block_nz || is_nonzero(source->at(row, col));
                }
            }
            if (block_nz) {
                col_idxs[block] = bcol;
                for (int lrow = 0; lrow < bs; ++lrow) {
                    for (int lcol = 0; lcol < bs; ++lcol) {
                        const auto row = lrow + bs * brow;
                        const auto col = lcol + bs * bcol;
                        blocks(block, lrow, lcol) = source->at(row, col);
                    }
                }
                block++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_FBCSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_hybrid(std::shared_ptr<const DefaultExecutor> exec,
                       const matrix::Dense<ValueType>* source,
                       const int64* coo_row_ptrs,
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto ell_lim = result->get_ell_num_stored_elements_per_row();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; row++) {
        size_type ell_count = 0;
        size_type col = 0;
        for (; col < num_cols && ell_count < ell_lim; col++) {
            auto val = source->at(row, col);
            if (is_nonzero(val)) {
                result->ell_val_at(row, ell_count) = val;
                result->ell_col_at(row, ell_count) = col;
                ell_count++;
            }
        }
        for (; ell_count < ell_lim; ell_count++) {
            result->ell_val_at(row, ell_count) = zero<ValueType>();
            result->ell_col_at(row, ell_count) = invalid_index<IndexType>();
        }
        auto coo_idx = coo_row_ptrs[row];
        for (; col < num_cols; col++) {
            auto val = source->at(row, col);
            if (is_nonzero(val)) {
                coo_val[coo_idx] = val;
                coo_col[coo_idx] = col;
                coo_row[coo_idx] = row;
                coo_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_HYBRID_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sellp(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Dense<ValueType>* source,
                      matrix::Sellp<ValueType, IndexType>* result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto vals = result->get_values();
    const auto col_idxs = result->get_col_idxs();
    const auto slice_sets = result->get_slice_sets();
    const auto slice_size = result->get_slice_size();
    const auto num_slices = ceildiv(num_rows, slice_size);
#pragma omp parallel for
    for (size_type slice = 0; slice < num_slices; slice++) {
        for (size_type local_row = 0; local_row < slice_size; local_row++) {
            const auto row = slice * slice_size + local_row;
            if (row >= num_rows) {
                break;
            }
            auto sellp_idx = slice_sets[slice] * slice_size + local_row;
            const auto sellp_end =
                slice_sets[slice + 1] * slice_size + local_row;
            for (size_type col = 0; col < num_cols; col++) {
                auto val = source->at(row, col);
                if (is_nonzero(val)) {
                    col_idxs[sellp_idx] = col;
                    vals[sellp_idx] = val;
                    sellp_idx += slice_size;
                }
            }
            for (; sellp_idx < sellp_end; sellp_idx += slice_size) {
                col_idxs[sellp_idx] = invalid_index<IndexType>();
                vals[sellp_idx] = zero<ValueType>();
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Dense<ValueType>* source,
                             matrix::SparsityCsr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto value = result->get_value();
    value[0] = one<ValueType>();

#pragma omp parallel for
    for (size_type row = 0; row < num_rows; ++row) {
        auto cur_ptr = row_ptrs[row];
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source->at(row, col);
            if (is_nonzero(val)) {
                col_idxs[cur_ptr] = col;
                ++cur_ptr;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void transpose(std::shared_ptr<const DefaultExecutor> exec,
               const matrix::Dense<ValueType>* orig,
               matrix::Dense<ValueType>* trans)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            trans->at(j, i) = orig->at(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Dense<ValueType>* orig,
                    matrix::Dense<ValueType>* trans)
{
#pragma omp parallel for
    for (size_type i = 0; i < orig->get_size()[0]; ++i) {
        for (size_type j = 0; j < orig->get_size()[1]; ++j) {
            trans->at(j, i) = conj(orig->at(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzero_blocks_per_row(std::shared_ptr<const DefaultExecutor> exec,
                                  const matrix::Dense<ValueType>* source,
                                  int bs, IndexType* result)
{
    const auto num_rows = source->get_size()[0];
    const auto num_cols = source->get_size()[1];
    const auto num_block_rows = num_rows / bs;
    const auto num_block_cols = num_cols / bs;
#pragma omp parallel for
    for (size_type brow = 0; brow < num_block_rows; ++brow) {
        IndexType num_nonzero_blocks{};
        for (size_type bcol = 0; bcol < num_block_cols; ++bcol) {
            bool block_nz = false;
            for (int lrow = 0; lrow < bs; ++lrow) {
                for (int lcol = 0; lcol < bs; ++lcol) {
                    const auto row = lrow + bs * brow;
                    const auto col = lcol + bs * bcol;
                    block_nz = block_nz || is_nonzero(source->at(row, col));
                }
            }
            num_nonzero_blocks += block_nz ? 1 : 0;
        }
        result[brow] = num_nonzero_blocks;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZERO_BLOCKS_PER_ROW_KERNEL);


}  // namespace dense
}  // namespace omp
}  // namespace kernels
}  // namespace gko
