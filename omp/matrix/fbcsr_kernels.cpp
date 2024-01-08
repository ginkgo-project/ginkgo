// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/fbcsr_kernels.hpp"


#include <algorithm>
#include <numeric>
#include <utility>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "accessor/block_col_major.hpp"
#include "core/base/allocator.hpp"
#include "core/base/block_sizes.hpp"
#include "core/base/device_matrix_data_kernels.hpp"
#include "core/base/iterator_factory.hpp"
#include "core/base/utils.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The fixed-block compressed sparse row matrix format namespace.
 *
 * @ingroup fbcsr
 */
namespace fbcsr {


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
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

#pragma omp parallel for
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
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
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

#pragma omp parallel for
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
        exec, data.get_num_stored_elements()};
    components::soa_to_aos(exec, data, block_ordered);
    const auto in_nnz = data.get_num_stored_elements();
    auto block_ordered_ptr = block_ordered.get_data();
    std::sort(
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
    while (block_row < static_cast<int64>(row_ptrs.get_size() - 1)) {
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
void fill_in_dense(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Fbcsr<ValueType, IndexType>* const source,
                   matrix::Dense<ValueType>* const result)
{
    const auto bs = source->get_block_size();
    const auto nbrows = source->get_num_block_rows();
    const auto nbnz = source->get_num_stored_blocks();
    auto row_ptrs = source->get_const_row_ptrs();
    auto col_idxs = source->get_const_col_idxs();
    const acc::range<acc::block_col_major<const ValueType, 3>> values{
        to_std_array<acc::size_type>(nbnz, bs, bs), source->get_const_values()};
#pragma omp parallel for
    for (size_type block_row = 0; block_row < nbrows; block_row++) {
        const auto row_begin = row_ptrs[block_row];
        const auto row_end = row_ptrs[block_row + 1];
        for (auto block = row_begin; block < row_end; block++) {
            const auto block_col = col_idxs[block];
            for (int local_row = 0; local_row < bs; local_row++) {
                const auto row = block_row * bs + local_row;
                for (int local_col = 0; local_col < bs; local_col++) {
                    const auto col = block_col * bs + local_col;
                    result->at(row, col) = values(block, local_row, local_col);
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(const std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Fbcsr<ValueType, IndexType>* const source,
                    matrix::Csr<ValueType, IndexType>* const result)
{
    const auto nbrows = source->get_num_block_rows();
    const auto bs = source->get_block_size();
    const auto block_row_ptrs = source->get_const_row_ptrs();
    const auto block_col_idxs = source->get_const_col_idxs();
    const auto row_ptrs = result->get_row_ptrs();
    const auto col_idxs = result->get_col_idxs();
    const auto vals = result->get_values();
    auto sizes =
        gko::to_std_array<acc::size_type>(block_row_ptrs[nbrows], bs, bs);
    const auto block_vals =
        acc::range<acc::block_col_major<const ValueType, 3>>(
            sizes, source->get_const_values());
#pragma omp parallel for
    for (IndexType block_row = 0; block_row < nbrows; block_row++) {
        const auto block_row_begin = block_row_ptrs[block_row];
        const auto block_row_end = block_row_ptrs[block_row + 1];
        const auto block_row_size = block_row_end - block_row_begin;
        const auto row_size = block_row_size * bs;
        const auto row_base = block_row_begin * bs * bs;
        for (int local_row = 0; local_row < bs; local_row++) {
            const auto row = block_row * bs + local_row;
            row_ptrs[row] = row_base + row_size * local_row;
            for (auto block = block_row_begin; block < block_row_end; block++) {
                const auto block_base =
                    row_ptrs[row] + bs * (block - block_row_begin);
                for (int local_col = 0; local_col < bs; local_col++) {
                    const auto col = block_col_idxs[block] * bs + local_col;
                    const auto out_idx = block_base + local_col;
                    col_idxs[out_idx] = col;
                    vals[out_idx] = block_vals(block, local_row, local_col);
                }
            }
        }
    }
    row_ptrs[result->get_size()[0]] = source->get_num_stored_elements();
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
    auto sizes =
        gko::to_std_array<acc::size_type>(row_ptrs[num_blk_rows], blksz, blksz);
    const acc::range<acc::block_col_major<const ValueType, 3>> rvalues(
        sizes, fbcsr_vals);
    acc::range<acc::block_col_major<ValueType, 3>> cvalues(sizes, csc_vals);
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
    std::shared_ptr<const OmpExecutor> exec,
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
void transpose(std::shared_ptr<const OmpExecutor> exec,
               const matrix::Fbcsr<ValueType, IndexType>* const orig,
               matrix::Fbcsr<ValueType, IndexType>* const trans)
{
    transpose_and_transform(exec, trans, orig,
                            [](const ValueType x) { return x; });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FBCSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void conj_transpose(std::shared_ptr<const OmpExecutor> exec,
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
    std::shared_ptr<const OmpExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType>* const to_check,
    bool* const is_sorted)
{
    const auto row_ptrs = to_check->get_const_row_ptrs();
    const auto col_idxs = to_check->get_const_col_idxs();
    const auto size = to_check->get_num_block_rows();
    bool local_is_sorted = true;
#pragma omp parallel for reduction(&& : local_is_sorted)
    for (size_type i = 0; i < size; ++i) {
        // Skip comparison if any thread detects that it is not sorted
        if (local_is_sorted) {
            for (auto idx = row_ptrs[i] + 1; idx < row_ptrs[i + 1]; ++idx) {
                if (col_idxs[idx - 1] > col_idxs[idx]) {
                    local_is_sorted = false;
                    break;
                }
            }
        }
    }
    *is_sorted = local_is_sorted;
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
#pragma omp parallel for
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
void sort_by_column_index(const std::shared_ptr<const OmpExecutor> exec,
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
void extract_diagonal(std::shared_ptr<const OmpExecutor> exec,
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

    const acc::range<acc::block_col_major<const ValueType, 3>> vblocks(
        gko::to_std_array<acc::size_type>(row_ptrs[nbrows], bs, bs), values);

#pragma omp parallel for
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
}  // namespace omp
}  // namespace kernels
}  // namespace gko
