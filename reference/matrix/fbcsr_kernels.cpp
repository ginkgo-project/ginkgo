/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>
#include <iterator>
#include <numeric>
#include <utility>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/blockutils.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/fixed_block.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"
#include "reference/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Compressed sparse row matrix format namespace.
 * @ref Fbcsr
 * @ingroup fbcsr
 */
namespace fbcsr {


template <typename ValueType, typename IndexType>
void spmv(const std::shared_ptr<const ReferenceExecutor> exec,
          const matrix::Fbcsr<ValueType, IndexType> *const a,
          const matrix::Dense<ValueType> *const b,
          matrix::Dense<ValueType> *const c)
{
    const int bs = a->get_block_size();
    const IndexType nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows = a->get_num_block_rows();
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    const blockutils::DenseBlocksView<const ValueType, IndexType> avalues(
        vals, bs, bs);

    ValueType *const cvals = c->get_values();

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        const IndexType crowblkend = (ibrow + 1) * bs * nvecs;
        for (IndexType i = ibrow * bs * nvecs; i < crowblkend; i++)
            cvals[i] = zero<ValueType>();

        for (IndexType inz = row_ptrs[ibrow]; inz < row_ptrs[ibrow + 1];
             ++inz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = ibrow * bs + ib;
                for (int jb = 0; jb < bs; jb++) {
                    const auto val = avalues(inz, ib, jb);
                    const auto col = col_idxs[inz] * bs + jb;
                    for (size_type j = 0; j < nvecs; ++j)
                        c->at(row, j) += val * b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(const std::shared_ptr<const ReferenceExecutor> exec,
                   const matrix::Dense<ValueType> *const alpha,
                   const matrix::Fbcsr<ValueType, IndexType> *const a,
                   const matrix::Dense<ValueType> *const b,
                   const matrix::Dense<ValueType> *const beta,
                   matrix::Dense<ValueType> *const c)
{
    const int bs = a->get_block_size();
    const IndexType nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows = a->get_num_block_rows();
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    const blockutils::DenseBlocksView<const ValueType, IndexType> avalues(
        vals, bs, bs);

    ValueType *const cvals = c->get_values();

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        const IndexType crowblkend = (ibrow + 1) * bs * nvecs;
        for (IndexType i = ibrow * bs * nvecs; i < crowblkend; i++)
            cvals[i] *= vbeta;

        for (IndexType inz = row_ptrs[ibrow]; inz < row_ptrs[ibrow + 1];
             ++inz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = ibrow * bs + ib;
                for (int jb = 0; jb < bs; jb++) {
                    const auto val = avalues(inz, ib, jb);
                    const auto col = col_idxs[inz] * bs + jb;
                    for (size_type j = 0; j < nvecs; ++j)
                        c->at(row, j) += valpha * val * b->at(col, j);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spgemm_insert_row(unordered_set<IndexType> &cols,
                       const matrix::Fbcsr<ValueType, IndexType> *c,
                       size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto row_ptrs = c->get_const_row_ptrs();
//    auto col_idxs = c->get_const_col_idxs();
//    cols.insert(col_idxs + row_ptrs[row], col_idxs + row_ptrs[row + 1]);
//}


template <typename ValueType, typename IndexType>
void spgemm_insert_row2(unordered_set<IndexType> &cols,
                        const matrix::Fbcsr<ValueType, IndexType> *a,
                        const matrix::Fbcsr<ValueType, IndexType> *b,
                        size_type row) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:fbcsr): change the code imported from matrix/csr if needed
//    auto a_row_ptrs = a->get_const_row_ptrs();
//    auto a_col_idxs = a->get_const_col_idxs();
//    auto b_row_ptrs = b->get_const_row_ptrs();
//    auto b_col_idxs = b->get_const_col_idxs();
//    for (size_type a_nz = a_row_ptrs[row];
//         a_nz < size_type(a_row_ptrs[row + 1]); ++a_nz) {
//        auto a_col = a_col_idxs[a_nz];
//        auto b_row = a_col;
//        cols.insert(b_col_idxs + b_row_ptrs[b_row],
//                    b_col_idxs + b_row_ptrs[b_row + 1]);
//    }
//}


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row(map<IndexType, ValueType> &cols,
                           const matrix::Fbcsr<ValueType, IndexType> *c,
                           ValueType scale, size_type row) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void spgemm_accumulate_row2(map<IndexType, ValueType> &cols,
                            const matrix::Fbcsr<ValueType, IndexType> *a,
                            const matrix::Fbcsr<ValueType, IndexType> *b,
                            ValueType scale, size_type row) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void spgemm(std::shared_ptr<const ReferenceExecutor> exec,
            const matrix::Fbcsr<ValueType, IndexType> *a,
            const matrix::Fbcsr<ValueType, IndexType> *b,
            matrix::Fbcsr<ValueType, IndexType> *c) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void advanced_spgemm(std::shared_ptr<const ReferenceExecutor> exec,
                     const matrix::Dense<ValueType> *alpha,
                     const matrix::Fbcsr<ValueType, IndexType> *a,
                     const matrix::Fbcsr<ValueType, IndexType> *b,
                     const matrix::Dense<ValueType> *beta,
                     const matrix::Fbcsr<ValueType, IndexType> *d,
                     matrix::Fbcsr<ValueType, IndexType> *c)
    GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_dense(const std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *const source,
                      matrix::Dense<ValueType> *const result)
{
    const int bs = source->get_block_size();
    const size_type nbrows = source->get_num_block_rows();
    const size_type nbcols = source->get_num_block_cols();
    const IndexType *const row_ptrs = source->get_const_row_ptrs();
    const IndexType *const col_idxs = source->get_const_col_idxs();
    const ValueType *const vals = source->get_const_values();

    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> values(
        vals, bs, bs);

    for (size_type brow = 0; brow < nbrows; ++brow) {
        for (size_type bcol = 0; bcol < nbcols; ++bcol) {
            for (int ib = 0; ib < bs; ib++)
                for (int jb = 0; jb < bs; jb++)
                    result->at(brow * bs + ib, bcol * bs + jb) =
                        zero<ValueType>();
        }
        for (IndexType ibnz = row_ptrs[brow]; ibnz < row_ptrs[brow + 1];
             ++ibnz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = brow * bs + ib;
                for (int jb = 0; jb < bs; jb++)
                    result->at(row, col_idxs[ibnz] * bs + jb) =
                        values(ibnz, ib, jb);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *const source,
                    matrix::Csr<ValueType, IndexType> *const result)
{
    const int bs = source->get_block_size();
    const size_type nbrows = source->get_num_block_rows();
    const size_type nbcols = source->get_num_block_cols();
    const IndexType *const browptrs = source->get_const_row_ptrs();
    const IndexType *const bcolinds = source->get_const_col_idxs();
    const ValueType *const bvals = source->get_const_values();

    assert(nbrows * bs == result->get_size()[0]);
    assert(nbcols * bs == result->get_size()[1]);
    assert(source->get_num_stored_elements() ==
           result->get_num_stored_elements());

    IndexType *const row_ptrs = result->get_row_ptrs();
    IndexType *const col_idxs = result->get_col_idxs();
    ValueType *const vals = result->get_values();

    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> bvalues(
        bvals, bs, bs);

    for (size_type brow = 0; brow < nbrows; ++brow) {
        const IndexType nz_browstart = browptrs[brow] * bs * bs;
        const IndexType numblocks_brow = browptrs[brow + 1] - browptrs[brow];
        for (int ib = 0; ib < bs; ib++)
            row_ptrs[brow * bs + ib] = nz_browstart + numblocks_brow * bs * ib;

        for (IndexType ibnz = browptrs[brow]; ibnz < browptrs[brow + 1];
             ++ibnz) {
            const IndexType bcol = bcolinds[ibnz];

            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = brow * bs + ib;
                const IndexType inz_blockstart =
                    row_ptrs[row] + (ibnz - browptrs[brow]) * bs;

                for (int jb = 0; jb < bs; jb++) {
                    const IndexType inz = inz_blockstart + jb;
                    vals[inz] = bvalues(ibnz, ib, jb);
                    col_idxs[inz] = bcol * bs + jb;
                }
            }
        }
    }

    row_ptrs[source->get_size()[0]] = source->get_num_stored_elements();
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator,
          bool transpose_blocks>
void convert_fbcsr_to_fbcsc(const size_type num_blk_rows, const int blksz,
                            const IndexType *const row_ptrs,
                            const IndexType *const col_idxs,
                            const ValueType *const fbcsr_vals,
                            IndexType *const row_idxs,
                            IndexType *const col_ptrs,
                            ValueType *const csc_vals, UnaryOperator op)
{
    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> rvalues(
        fbcsr_vals, blksz, blksz);
    gko::blockutils::DenseBlocksView<ValueType, IndexType> cvalues(
        csc_vals, blksz, blksz);
    for (size_type brow = 0; brow < num_blk_rows; ++brow) {
        for (auto i = row_ptrs[brow]; i < row_ptrs[brow + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = brow;
            for (int ib = 0; ib < blksz; ib++)
                for (int jb = 0; jb < blksz; jb++)
                    // csc_vals[dest_idx] = op(fbcsr_vals[i]);
                    cvalues(dest_idx, ib, jb) =
                        op(transpose_blocks ? rvalues(i, jb, ib)
                                            : rvalues(i, ib, jb));
        }
    }
}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(
    const std::shared_ptr<const ReferenceExecutor> exec,
    matrix::Fbcsr<ValueType, IndexType> *const trans,
    const matrix::Fbcsr<ValueType, IndexType> *const orig, UnaryOperator op)
{
    const int bs = orig->get_block_size();
    auto trans_row_ptrs = trans->get_row_ptrs();
    auto orig_row_ptrs = orig->get_const_row_ptrs();
    auto trans_col_idxs = trans->get_col_idxs();
    auto orig_col_idxs = orig->get_const_col_idxs();
    auto trans_vals = trans->get_values();
    auto orig_vals = orig->get_const_values();

    auto orig_num_cols = orig->get_size()[1];
    const size_type nbcols = orig->get_num_block_cols();
    auto orig_num_rows = orig->get_size()[0];
    const size_type nbrows = orig->get_num_block_rows();
    auto orig_nbnz = orig_row_ptrs[nbrows];

    trans_row_ptrs[0] = 0;
    convert_idxs_to_ptrs(orig_col_idxs, orig_nbnz, trans_row_ptrs + 1, nbcols);

    convert_fbcsr_to_fbcsc<ValueType, IndexType, UnaryOperator, true>(
        nbrows, bs, orig_row_ptrs, orig_col_idxs, orig_vals, trans_col_idxs,
        trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType> *const orig,
               matrix::Fbcsr<ValueType, IndexType> *const trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType> *const orig,
                    matrix::Fbcsr<ValueType, IndexType> *const trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const source,
    size_type *const result)
{
    const auto num_rows = source->get_size()[0];
    const auto row_ptrs = source->get_const_row_ptrs();
    const int bs = source->get_block_size();
    IndexType max_nnz = 0;

    for (size_type i = 0; i < num_rows; i++) {
        const size_type ibrow = i / bs;
        max_nnz =
            std::max((row_ptrs[ibrow + 1] - row_ptrs[ibrow]) * bs, max_nnz);
    }

    *result = max_nnz;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_nonzeros_per_row(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *source, Array<size_type> *result)
{
    const auto row_ptrs = source->get_const_row_ptrs();
    auto row_nnz_val = result->get_data();
    const int bs = source->get_block_size();
    assert(result->get_num_elems() == source->get_size()[0]);

    for (size_type i = 0; i < result->get_num_elems(); i++) {
        const size_type ibrow = i / bs;
        row_nnz_val[i] = (row_ptrs[ibrow + 1] - row_ptrs[ibrow]) * bs;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CALCULATE_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType> *to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const to_check,
    bool *const is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_size();
    const int bs = to_check->get_block_size();
    const size_type nbrows = to_check->get_num_block_rows();

    for (size_type i = 0; i < nbrows; ++i) {
        for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
            if (col_idxs[idx - 1] > col_idxs[idx]) {
                *is_sorted = false;
                return;
            }
        }
    }
    *is_sorted = true;
    return;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_IS_SORTED_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      const matrix::Fbcsr<ValueType, IndexType> *const orig,
                      matrix::Diagonal<ValueType> *const diag)
{
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const int bs = orig->get_block_size();
    const size_type diag_size = diag->get_size()[0];
    const size_type nbrows = orig->get_num_block_rows();
    auto diag_values = diag->get_values();
    assert(diag_size == orig->get_size()[0]);

    const gko::blockutils::DenseBlocksView<const ValueType, IndexType> vblocks(
        values, bs, bs);

    for (size_type ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (size_type idx = row_ptrs[ibrow]; idx < row_ptrs[ibrow + 1];
             ++idx) {
            if (col_idxs[idx] == ibrow) {
                for (int ib = 0; ib < bs; ib++)
                    diag_values[ibrow * bs + ib] = vblocks(idx, ib, ib);
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_EXTRACT_DIAGONAL);


}  // namespace fbcsr
}  // namespace reference
}  // namespace kernels
}  // namespace gko
