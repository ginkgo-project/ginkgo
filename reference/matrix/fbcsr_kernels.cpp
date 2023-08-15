// SPDX-FileCopyrightText: 2017-2023 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

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
#include "core/base/allocator.hpp"
#include "core/base/block_sizes.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/fbcsr_builder.hpp"
#include "core/synthesizer/implementation_selection.hpp"


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
          const matrix::Fbcsr<ValueType, IndexType>* const a,
          const matrix::Dense<ValueType>* const b,
          matrix::Dense<ValueType>* const c)
{
    const int bs = a->get_block_size();
    const auto nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows = a->get_num_block_rows();
    const size_type nbnz = a->get_num_stored_blocks();
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    const acc::range<acc::block_col_major<const ValueType, 3>> avalues{
        to_std_array<acc::size_type>(nbnz, bs, bs), a->get_const_values()};

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (IndexType row = ibrow * bs; row < (ibrow + 1) * bs; ++row) {
            for (IndexType rhs = 0; rhs < nvecs; rhs++) {
                c->at(row, rhs) = zero<ValueType>();
            }
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
                   const matrix::Dense<ValueType>* const alpha,
                   const matrix::Fbcsr<ValueType, IndexType>* const a,
                   const matrix::Dense<ValueType>* const b,
                   const matrix::Dense<ValueType>* const beta,
                   matrix::Dense<ValueType>* const c)
{
    const int bs = a->get_block_size();
    const auto nvecs = static_cast<IndexType>(b->get_size()[1]);
    const IndexType nbrows = a->get_num_block_rows();
    const size_type nbnz = a->get_num_stored_blocks();
    auto row_ptrs = a->get_const_row_ptrs();
    auto col_idxs = a->get_const_col_idxs();
    auto valpha = alpha->at(0, 0);
    auto vbeta = beta->at(0, 0);
    const acc::range<acc::block_col_major<const ValueType, 3>> avalues{
        to_std_array<acc::size_type>(nbnz, bs, bs), a->get_const_values()};

    for (IndexType ibrow = 0; ibrow < nbrows; ++ibrow) {
        for (IndexType row = ibrow * bs; row < (ibrow + 1) * bs; ++row) {
            for (IndexType rhs = 0; rhs < nvecs; rhs++) {
                c->at(row, rhs) *= vbeta;
            }
        }

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
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         device_matrix_data<ValueType, IndexType>& data,
                         int block_size, array<IndexType>& row_ptrs,
                         array<IndexType>& col_idxs, array<ValueType>& values)
{
    array<matrix_data_entry<ValueType, IndexType>> block_ordered{
        exec, data.get_num_elems()};
    components::soa_to_aos(exec, data, block_ordered);
    const auto in_nnz = data.get_num_elems();
    auto block_ordered_ptr = block_ordered.get_data();
    std::stable_sort(
        block_ordered_ptr, block_ordered_ptr + in_nnz,
        [block_size](auto a, auto b) {
            return std::make_tuple(a.row / block_size, a.column / block_size) <
                   std::make_tuple(b.row / block_size, b.column / block_size);
        });
    auto row_ptrs_ptr = row_ptrs.get_data();
    gko::vector<IndexType> col_idx_vec{{exec}};
    gko::vector<ValueType> value_vec{{exec}};
    int64 block_row = -1;
    int64 block_col = -1;
    for (size_type i = 0; i < in_nnz; i++) {
        const auto entry = block_ordered_ptr[i];
        const auto new_block_row = entry.row / block_size;
        const auto new_block_col = entry.column / block_size;
        while (new_block_row > block_row) {
            // we finished row block_row, so store its end pointer
            row_ptrs_ptr[block_row + 1] = col_idx_vec.size();
            block_col = -1;
            ++block_row;
        }
        if (new_block_col != block_col) {
            // we encountered a new column, so insert it with block storage
            col_idx_vec.emplace_back(new_block_col);
            value_vec.resize(value_vec.size() + block_size * block_size);
            block_col = new_block_col;
        }
        const auto local_row = entry.row % block_size;
        const auto local_col = entry.column % block_size;
        value_vec[value_vec.size() - block_size * block_size + local_row +
                  local_col * block_size] = entry.value;
    }
    while (block_row < static_cast<int64>(row_ptrs.get_num_elems() - 1)) {
        // we finished row block_row, so store its end pointer
        row_ptrs_ptr[block_row + 1] = col_idx_vec.size();
        ++block_row;
    }
    values.resize_and_reset(value_vec.size());
    col_idxs.resize_and_reset(col_idx_vec.size());
    std::copy(value_vec.begin(), value_vec.end(), values.get_data());
    std::copy(col_idx_vec.begin(), col_idx_vec.end(), col_idxs.get_data());
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(const std::shared_ptr<const ReferenceExecutor>,
                   const matrix::Fbcsr<ValueType, IndexType>* const source,
                   matrix::Dense<ValueType>* const result)
{
    const int bs = source->get_block_size();
    const IndexType nbrows = source->get_num_block_rows();
    const IndexType nbcols = source->get_num_block_cols();
    const IndexType* const row_ptrs = source->get_const_row_ptrs();
    const IndexType* const col_idxs = source->get_const_col_idxs();
    const ValueType* const vals = source->get_const_values();

    const acc::range<acc::block_col_major<const ValueType, 3>> values{
        std::array<acc::size_type, 3>{
            static_cast<acc::size_type>(source->get_num_stored_blocks()),
            static_cast<acc::size_type>(bs), static_cast<acc::size_type>(bs)},
        vals};

    for (IndexType brow = 0; brow < nbrows; ++brow) {
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
    GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const ReferenceExecutor>,
                    const matrix::Fbcsr<ValueType, IndexType>* const source,
                    matrix::Csr<ValueType, IndexType>* const result)
{
    const int bs = source->get_block_size();
    const IndexType nbrows = source->get_num_block_rows();
    const IndexType nbcols = source->get_num_block_cols();
    const IndexType* const browptrs = source->get_const_row_ptrs();
    const IndexType* const bcolinds = source->get_const_col_idxs();
    const ValueType* const bvals = source->get_const_values();

    assert(nbrows * bs == result->get_size()[0]);
    assert(nbcols * bs == result->get_size()[1]);
    assert(source->get_num_stored_elements() ==
           result->get_num_stored_elements());

    IndexType* const row_ptrs = result->get_row_ptrs();
    IndexType* const col_idxs = result->get_col_idxs();
    ValueType* const vals = result->get_values();

    const acc::range<acc::block_col_major<const ValueType, 3>> bvalues{
        std::array<acc::size_type, 3>{
            static_cast<acc::size_type>(source->get_num_stored_blocks()),
            static_cast<acc::size_type>(bs), static_cast<acc::size_type>(bs)},
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
                            const IndexType* const row_ptrs,
                            const IndexType* const col_idxs,
                            const ValueType* const fbcsr_vals,
                            IndexType* const row_idxs,
                            IndexType* const col_ptrs,
                            ValueType* const csc_vals, UnaryOperator op)
{
    const acc::range<acc::block_col_major<const ValueType, 3>> rvalues{
        std::array<acc::size_type, 3>{
            static_cast<acc::size_type>(row_ptrs[num_blk_rows]),
            static_cast<acc::size_type>(blksz),
            static_cast<acc::size_type>(blksz)},
        fbcsr_vals};
    const acc::range<acc::block_col_major<ValueType, 3>> cvalues{
        std::array<acc::size_type, 3>{
            static_cast<acc::size_type>(row_ptrs[num_blk_rows]),
            static_cast<acc::size_type>(blksz),
            static_cast<acc::size_type>(blksz)},
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
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::Fbcsr<ValueType, IndexType>* const trans,
    const matrix::Fbcsr<ValueType, IndexType>* const orig, UnaryOperator op)
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

    components::fill_array(exec, trans_row_ptrs, nbcols + 1, IndexType{});
    for (size_type i = 0; i < orig_nbnz; i++) {
        trans_row_ptrs[orig_col_idxs[i] + 1]++;
    }
    components::prefix_sum_nonnegative(exec, trans_row_ptrs + 1, nbcols);

    convert_fbcsr_to_fbcsc<ValueType, IndexType, UnaryOperator, true>(
        nbrows, bs, orig_row_ptrs, orig_col_idxs, orig_vals, trans_col_idxs,
        trans_row_ptrs + 1, trans_vals, op);
}


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType>* const orig,
               matrix::Fbcsr<ValueType, IndexType>* const trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* const orig,
                    matrix::Fbcsr<ValueType, IndexType>* const trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return conj(x); });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const ReferenceExecutor>,
    const matrix::Fbcsr<ValueType, IndexType>* const to_check,
    bool* const is_sorted)
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


namespace {


template <int mat_blk_sz, typename ValueType, typename IndexType>
void sort_by_column_index_impl(
    syn::value_list<int, mat_blk_sz>,
    matrix::Fbcsr<ValueType, IndexType>* const to_sort)
{
    auto row_ptrs = to_sort->get_const_row_ptrs();
    auto col_idxs = to_sort->get_col_idxs();
    auto values = to_sort->get_values();
    const auto nbrows = to_sort->get_num_block_rows();
    constexpr int bs2 = mat_blk_sz * mat_blk_sz;
    for (IndexType irow = 0; irow < nbrows; ++irow) {
        IndexType* const brow_col_idxs = col_idxs + row_ptrs[irow];
        ValueType* const brow_vals = values + row_ptrs[irow] * bs2;
        const IndexType nbnz_brow = row_ptrs[irow + 1] - row_ptrs[irow];

        std::vector<IndexType> col_permute(nbnz_brow);
        std::iota(col_permute.begin(), col_permute.end(), 0);
        auto it = detail::make_zip_iterator(brow_col_idxs, col_permute.data());
        std::sort(it, it + nbnz_brow, [](auto a, auto b) {
            return std::get<0>(a) < std::get<0>(b);
        });

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

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_sort_col_idx,
                                    sort_by_column_index_impl);


}  // namespace


template <typename ValueType, typename IndexType>
void sort_by_column_index(const std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::Fbcsr<ValueType, IndexType>* const to_sort)
{
    const int bs = to_sort->get_block_size();
    select_sort_col_idx(
        fixedblock::compiled_kernels(),
        [bs](int compiled_block_size) { return bs == compiled_block_size; },
        syn::value_list<int>(), syn::type_list<>(), to_sort);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor>,
                      const matrix::Fbcsr<ValueType, IndexType>* const orig,
                      matrix::Diagonal<ValueType>* const diag)
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
        std::array<acc::size_type, 3>{
            static_cast<acc::size_type>(orig->get_num_stored_blocks()),
            static_cast<acc::size_type>(bs), static_cast<acc::size_type>(bs)},
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
