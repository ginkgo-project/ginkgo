/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "accessor/block_col_major.hpp"
#include "accessor/range.hpp"
#include "core/base/allocator.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"
#include "reference/components/format_conversion.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The fixed-block compressed sparse row matrix format namespace.
 *
 * @ingroup fbcsr
 */
namespace fbcsr {


template <typename ValueType, typename IndexType>
void spmv(const std::shared_ptr<const ReferenceExecutor>,
          const matrix::Fbcsr<ValueType, IndexType> *const a,
          const matrix::Dense<ValueType> *const b,
          matrix::Dense<ValueType> *const c)
{
    const int bs = a->get_block_size();
    const auto nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows = a->get_num_block_rows();
    const size_type nbnz = a->get_num_stored_blocks();
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    const acc::range<acc::block_col_major<const ValueType, 3>> avalues{
        std::array<size_type, 3>{nbnz, static_cast<size_type>(bs),
                                 static_cast<size_type>(bs)},
        vals};

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (IndexType i = ibrow * bs * nvecs; i < (ibrow + 1) * bs * nvecs;
             ++i) {
            c->get_values()[i] = zero<ValueType>();
        }
        for (IndexType inz = row_ptrs[ibrow]; inz < row_ptrs[ibrow + 1];
             ++inz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = ibrow * bs + ib;
                for (int jb = 0; jb < bs; jb++) {
                    const auto val = avalues(inz, ib, jb);
                    const auto col = col_idxs[inz] * bs + jb;
                    for (size_type j = 0; j < nvecs; ++j) {
                        c->at(row, j) += val * b->at(col, j);
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_FBCSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(const std::shared_ptr<const ReferenceExecutor>,
                   const matrix::Dense<ValueType> *const alpha,
                   const matrix::Fbcsr<ValueType, IndexType> *const a,
                   const matrix::Dense<ValueType> *const b,
                   const matrix::Dense<ValueType> *const beta,
                   matrix::Dense<ValueType> *const c)
{
    const int bs = a->get_block_size();
    const auto nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows = a->get_num_block_rows();
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto vals = a->get_const_values();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    const acc::range<acc::block_col_major<const ValueType, 3>> avalues{
        std::array<size_type, 3>{a->get_num_stored_blocks(),
                                 static_cast<size_type>(bs),
                                 static_cast<size_type>(bs)},
        vals};

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (IndexType i = ibrow * bs * nvecs; i < (ibrow + 1) * bs * nvecs;
             ++i)
            c->get_values()[i] *= vbeta;

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
void convert_to_dense(const std::shared_ptr<const ReferenceExecutor>,
                      const matrix::Fbcsr<ValueType, IndexType> *const source,
                      matrix::Dense<ValueType> *const result)
{
    const int bs = source->get_block_size();
    const IndexType nbrows = source->get_num_block_rows();
    const IndexType nbcols = source->get_num_block_cols();
    const IndexType *const row_ptrs = source->get_const_row_ptrs();
    const IndexType *const col_idxs = source->get_const_col_idxs();
    const ValueType *const vals = source->get_const_values();

    const acc::range<acc::block_col_major<const ValueType, 3>> values{
        std::array<size_type, 3>{source->get_num_stored_blocks(),
                                 static_cast<size_type>(bs),
                                 static_cast<size_type>(bs)},
        vals};

    for (IndexType brow = 0; brow < nbrows; ++brow) {
        for (size_type bcol = 0; bcol < nbcols; ++bcol) {
            for (int ib = 0; ib < bs; ib++) {
                for (int jb = 0; jb < bs; jb++) {
                    result->at(brow * bs + ib, bcol * bs + jb) =
                        zero<ValueType>();
                }
            }
        }
        for (IndexType ibnz = row_ptrs[brow]; ibnz < row_ptrs[brow + 1];
             ++ibnz) {
            for (int ib = 0; ib < bs; ib++) {
                const IndexType row = brow * bs + ib;
                for (int jb = 0; jb < bs; jb++) {
                    result->at(row, col_idxs[ibnz] * bs + jb) =
                        values(ibnz, ib, jb);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const ReferenceExecutor>,
                    const matrix::Fbcsr<ValueType, IndexType> *const source,
                    matrix::Csr<ValueType, IndexType> *const result)
{
    const int bs = source->get_block_size();
    const IndexType nbrows = source->get_num_block_rows();
    const IndexType nbcols = source->get_num_block_cols();
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

    const acc::range<acc::block_col_major<const ValueType, 3>> bvalues{
        std::array<size_type, 3>{source->get_num_stored_blocks(),
                                 static_cast<size_type>(bs),
                                 static_cast<size_type>(bs)},
        bvals};

    for (IndexType brow = 0; brow < nbrows; ++brow) {
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

    row_ptrs[source->get_size()[0]] =
        static_cast<IndexType>(source->get_num_stored_elements());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType, typename UnaryOperator,
          bool transpose_blocks>
void convert_fbcsr_to_fbcsc(const IndexType num_blk_rows, const int blksz,
                            const IndexType *const row_ptrs,
                            const IndexType *const col_idxs,
                            const ValueType *const fbcsr_vals,
                            IndexType *const row_idxs,
                            IndexType *const col_ptrs,
                            ValueType *const csc_vals, UnaryOperator op)
{
    const acc::range<acc::block_col_major<const ValueType, 3>> rvalues{
        std::array<size_type, 3>{static_cast<size_type>(row_ptrs[num_blk_rows]),
                                 static_cast<size_type>(blksz),
                                 static_cast<size_type>(blksz)},
        fbcsr_vals};
    const acc::range<acc::block_col_major<ValueType, 3>> cvalues{
        std::array<size_type, 3>{static_cast<size_type>(row_ptrs[num_blk_rows]),
                                 static_cast<size_type>(blksz),
                                 static_cast<size_type>(blksz)},
        csc_vals};
    for (IndexType brow = 0; brow < num_blk_rows; ++brow) {
        for (auto i = row_ptrs[brow]; i < row_ptrs[brow + 1]; ++i) {
            const auto dest_idx = col_ptrs[col_idxs[i]];
            col_ptrs[col_idxs[i]]++;
            row_idxs[dest_idx] = brow;
            for (int ib = 0; ib < blksz; ib++) {
                for (int jb = 0; jb < blksz; jb++) {
                    cvalues(dest_idx, ib, jb) =
                        op(transpose_blocks ? rvalues(i, jb, ib)
                                            : rvalues(i, ib, jb));
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType, typename UnaryOperator>
void transpose_and_transform(
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

    const IndexType nbcols = orig->get_num_block_cols();
    const IndexType nbrows = orig->get_num_block_rows();
    auto orig_nbnz = orig_row_ptrs[nbrows];

    trans_row_ptrs[0] = 0;
    convert_idxs_to_ptrs(orig_col_idxs, orig_nbnz, trans_row_ptrs + 1, nbcols);

    convert_fbcsr_to_fbcsc<ValueType, IndexType, UnaryOperator, true>(
        nbrows, bs, orig_row_ptrs, orig_col_idxs, orig_vals, trans_col_idxs,
        trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor>,
               const matrix::Fbcsr<ValueType, IndexType> *const orig,
               matrix::Fbcsr<ValueType, IndexType> *const trans)
{
    transpose_and_transform(trans, orig, [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor>,
                    const matrix::Fbcsr<ValueType, IndexType> *const orig,
                    matrix::Fbcsr<ValueType, IndexType> *const trans)
{
    transpose_and_transform(trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void calculate_max_nnz_per_row(
    std::shared_ptr<const ReferenceExecutor>,
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
    std::shared_ptr<const ReferenceExecutor>,
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
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor>,
    const matrix::Fbcsr<ValueType, IndexType> *const to_check,
    bool *const is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
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


template <int mat_blk_sz, typename ValueType, typename IndexType>
static void sort_by_column_index_impl(
    matrix::Fbcsr<ValueType, IndexType> *const to_sort)
{
    auto row_ptrs = to_sort->get_const_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    auto values = to_sort->get_values();
    const auto nbrows = to_sort->get_num_block_rows();
    constexpr int bs2 = mat_blk_sz * mat_blk_sz;
    for (IndexType i = 0; i < nbrows; ++i) {
        IndexType *const brow_col_idxs = col_idxs + row_ptrs[i];
        ValueType *const brow_vals = values + row_ptrs[i] * bs2;
        const IndexType nbnz_brow = row_ptrs[i + 1] - row_ptrs[i];

        std::vector<IndexType> col_permute(nbnz_brow);
        std::iota(col_permute.begin(), col_permute.end(), 0);
        auto helper = detail::IteratorFactory<IndexType, IndexType>(
            brow_col_idxs, col_permute.data(), nbnz_brow);
        std::sort(helper.begin(), helper.end());

        std::vector<ValueType> oldvalues(nbnz_brow * bs2);
        std::copy(brow_vals, brow_vals + nbnz_brow * bs2, oldvalues.begin());
        for (IndexType ibz = 0; ibz < nbnz_brow; ibz++) {
            for (int i = 0; i < bs2; i++) {
                brow_vals[ibz * bs2 + i] =
                    oldvalues[col_permute[ibz] * bs2 + i];
            }
        }
    }
}

template <typename ValueType, typename IndexType>
void sort_by_column_index(const std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType> *const to_sort)
{
    const int bs = to_sort->get_block_size();
    if (bs == 2) {
        sort_by_column_index_impl<2>(to_sort);
    } else if (bs == 3) {
        sort_by_column_index_impl<3>(to_sort);
    } else if (bs == 4) {
        sort_by_column_index_impl<4>(to_sort);
    } else {
        GKO_NOT_IMPLEMENTED;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor>,
                      const matrix::Fbcsr<ValueType, IndexType> *const orig,
                      matrix::Diagonal<ValueType> *const diag)
{
    const auto row_ptrs = orig->get_const_row_ptrs();
    const auto col_idxs = orig->get_const_col_idxs();
    const auto values = orig->get_const_values();
    const int bs = orig->get_block_size();
    const IndexType nbrows = orig->get_num_block_rows();
    const IndexType nbdim_min =
        std::min(orig->get_num_block_rows(), orig->get_num_block_cols());
    auto diag_values = diag->get_values();

    assert(diag->get_size()[0] == nbdim_min * bs);

    const acc::range<acc::block_col_major<const ValueType, 3>> vblocks{
        std::array<size_type, 3>{orig->get_num_stored_blocks(),
                                 static_cast<size_type>(bs),
                                 static_cast<size_type>(bs)},
        values};

    for (IndexType ibrow = 0; ibrow < nbdim_min; ++ibrow) {
        for (IndexType idx = row_ptrs[ibrow]; idx < row_ptrs[ibrow + 1];
             ++idx) {
            if (col_idxs[idx] == ibrow) {
                for (int ib = 0; ib < bs; ib++) {
                    diag_values[ibrow * bs + ib] = vblocks(idx, ib, ib);
                }
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
