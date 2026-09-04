// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include "accessor/block_col_major.hpp"
#include "core/components/prefix_sum_kernels.hpp"

namespace gko {
namespace kernels {
namespace reference {
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> a,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> c)
{
    for (size_type row = 0; row < c.size[0]; ++row) {
        for (size_type col = 0; col < c.size[1]; ++col) {
            c(row, col) = zero<ValueType>();
        }
    }

    for (size_type row = 0; row < c.size[0]; ++row) {
        for (size_type inner = 0; inner < a.size[1]; ++inner) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) += a(row, inner) * b(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           matrix::view::dense<const ValueType> alpha,
           matrix::view::dense<const ValueType> a,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<const ValueType> beta,
           matrix::view::dense<ValueType> c)
{
    if (is_nonzero(beta(0, 0))) {
        for (size_type row = 0; row < c.size[0]; ++row) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) *= beta(0, 0);
            }
        }
    } else {
        for (size_type row = 0; row < c.size[0]; ++row) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) = zero<ValueType>();
            }
        }
    }

    for (size_type row = 0; row < c.size[0]; ++row) {
        for (size_type inner = 0; inner < a.size[1]; ++inner) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) += alpha(0, 0) * a(row, inner) * b(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::dense<const ValueType> source, const int64*,
                    matrix::view::coo<ValueType, IndexType> result)
{
    auto num_rows = result.size[0];
    auto num_cols = result.size[1];
    auto num_nonzeros = result.num_stored_elements;

    auto row_idxs = result.row_idxs;
    auto col_idxs = result.col_idxs;
    auto values = result.values;

    size_type idxs = 0;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source(row, col);
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
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::dense<const ValueType> source,
                    matrix::view::csr<ValueType, IndexType> result)
{
    auto num_rows = result.size[0];
    auto num_cols = result.size[1];
    auto num_nonzeros = result.num_stored_elements;

    auto row_ptrs = result.row_ptrs;
    auto col_idxs = result.col_idxs;
    auto values = result.values;

    size_type cur_ptr = 0;
    row_ptrs[0] = cur_ptr;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source(row, col);
            if (is_nonzero(val)) {
                col_idxs[cur_ptr] = col;
                values[cur_ptr] = val;
                ++cur_ptr;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_ell(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::dense<const ValueType> source,
                    matrix::view::ell<ValueType, IndexType> result)
{
    auto num_rows = result.size[0];
    auto num_cols = result.size[1];
    auto max_nnz_per_row = result.num_stored_elements_per_row;
    for (size_type i = 0; i < max_nnz_per_row; i++) {
        for (size_type j = 0; j < num_rows; j++) {
            result.val_at(j, i) = zero<ValueType>();
            result.col_at(j, i) = invalid_index<IndexType>();
        }
    }
    size_type col_idx = 0;
    for (size_type row = 0; row < num_rows; row++) {
        col_idx = 0;
        for (size_type col = 0; col < num_cols; col++) {
            auto val = source(row, col);
            if (is_nonzero(val)) {
                result.val_at(row, col_idx) = val;
                result.col_at(row, col_idx) = col;
                col_idx++;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_ELL_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_fbcsr(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::view::dense<const ValueType> source,
                      matrix::Fbcsr<ValueType, IndexType>* result)
{
    const auto num_rows = source.size[0];
    const auto num_cols = source.size[1];
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
    for (size_type brow = 0; brow < num_block_rows; ++brow) {
        auto block = result->get_const_row_ptrs()[brow];
        for (size_type bcol = 0; bcol < num_block_cols; ++bcol) {
            bool block_nz = false;
            for (int lrow = 0; lrow < bs; ++lrow) {
                for (int lcol = 0; lcol < bs; ++lcol) {
                    const auto row = lrow + bs * brow;
                    const auto col = lcol + bs * bcol;
                    block_nz = block_nz || is_nonzero(source(row, col));
                }
            }
            if (block_nz) {
                col_idxs[block] = bcol;
                for (int lrow = 0; lrow < bs; ++lrow) {
                    for (int lcol = 0; lcol < bs; ++lcol) {
                        const auto row = lrow + bs * brow;
                        const auto col = lcol + bs * bcol;
                        blocks(block, lrow, lcol) = source(row, col);
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
void convert_to_hybrid(std::shared_ptr<const ReferenceExecutor> exec,
                       matrix::view::dense<const ValueType> source,
                       const int64*,
                       matrix::view::hybrid<ValueType, IndexType> result)
{
    auto num_rows = result.size[0];
    auto num_cols = result.size[1];
    auto ell_lim = result.ell_part.num_stored_elements_per_row;
    auto coo_lim = result.coo_part.num_stored_elements;
    auto coo_val = result.coo_part.values;
    auto coo_col = result.coo_part.col_idxs;
    auto coo_row = result.coo_part.row_idxs;
    std::fill_n(
        result.ell_part.values,
        result.ell_part.stride * result.ell_part.num_stored_elements_per_row,
        zero<ValueType>());
    std::fill_n(
        result.ell_part.col_idxs,
        result.ell_part.stride * result.ell_part.num_stored_elements_per_row,
        invalid_index<IndexType>());

    size_type coo_idx = 0;
    for (size_type row = 0; row < num_rows; row++) {
        size_type col = 0;
        for (size_type col_idx = 0; col < num_cols && col_idx < ell_lim;
             col++) {
            auto val = source(row, col);
            if (is_nonzero(val)) {
                result.ell_part.val_at(row, col_idx) = val;
                result.ell_part.col_at(row, col_idx) = col;
                col_idx++;
            }
        }
        for (; col < num_cols; col++) {
            auto val = source(row, col);
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
void convert_to_sellp(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::view::dense<const ValueType> source,
                      matrix::view::sellp<ValueType, IndexType> result)
{
    auto num_rows = result.size[0];
    auto num_cols = result.size[1];
    auto vals = result.values;
    auto col_idxs = result.col_idxs;
    auto slice_lengths = result.slice_lengths;
    auto slice_sets = result.slice_sets;
    auto slice_size = result.slice_size;
    for (size_type row = 0; row < num_rows; row++) {
        const auto slice = row / slice_size;
        const auto local_row = row % slice_size;
        auto sellp_ind = slice_sets[slice] * slice_size + local_row;
        const auto sellp_end = slice_sets[slice + 1] * slice_size + local_row;
        for (size_type col = 0; col < num_cols; col++) {
            auto val = source(row, col);
            if (is_nonzero(val)) {
                col_idxs[sellp_ind] = col;
                vals[sellp_ind] = val;
                sellp_ind += slice_size;
            }
        }
        for (; sellp_ind < sellp_end; sellp_ind += slice_size) {
            col_idxs[sellp_ind] = invalid_index<IndexType>();
            vals[sellp_ind] = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SELLP_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_sparsity_csr(std::shared_ptr<const ReferenceExecutor> exec,
                             matrix::view::dense<const ValueType> source,
                             matrix::SparsityCsr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto value = result->get_value();
    value[0] = one<ValueType>();
    size_type cur_ptr = 0;
    row_ptrs[0] = cur_ptr;
    for (size_type row = 0; row < num_rows; ++row) {
        for (size_type col = 0; col < num_cols; ++col) {
            auto val = source(row, col);
            if (is_nonzero(val)) {
                col_idxs[cur_ptr] = col;
                ++cur_ptr;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_CONVERT_TO_SPARSITY_CSR_KERNEL);


template <typename ValueType>
void compute_max_nnz_per_row(std::shared_ptr<const DefaultExecutor> exec,
                             matrix::view::dense<const ValueType> source,
                             size_type& result)
{
    auto num_rows = source.size[0];
    auto num_cols = source.size[1];
    result = 0;
    for (size_type row = 0; row < num_rows; ++row) {
        size_type num_nonzeros = 0;
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += is_nonzero(source(row, col));
        }
        result = std::max(num_nonzeros, result);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_MAX_NNZ_PER_ROW_KERNEL);


template <typename ValueType>
void compute_slice_sets(std::shared_ptr<const DefaultExecutor> exec,
                        matrix::view::dense<const ValueType> source,
                        size_type slice_size, size_type stride_factor,
                        size_type* slice_sets, size_type* slice_lengths)
{
    const auto num_rows = source.size[0];
    const auto num_cols = source.size[1];
    const auto num_slices = ceildiv(num_rows, slice_size);
    for (size_type slice = 0; slice < num_slices; slice++) {
        size_type slice_length = 0;
        for (size_type local_row = 0; local_row < slice_size; local_row++) {
            const auto row = slice * slice_size + local_row;
            size_type row_nnz{};
            if (row < num_rows) {
                for (size_type col = 0; col < num_cols; col++) {
                    row_nnz += is_nonzero(source(row, col));
                }
            }
            slice_length = std::max<size_type>(
                slice_length, ceildiv(row_nnz, stride_factor) * stride_factor);
        }
        slice_lengths[slice] = slice_length;
    }
    exec->copy(num_slices, slice_lengths, slice_sets);
    components::prefix_sum_nonnegative(exec, slice_sets, num_slices + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_SLICE_SETS_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                            matrix::view::dense<const ValueType> source,
                            IndexType* result)
{
    auto num_rows = source.size[0];
    auto num_cols = source.size[1];
    for (size_type row = 0; row < num_rows; ++row) {
        IndexType num_nonzeros{};
        for (size_type col = 0; col < num_cols; ++col) {
            num_nonzeros += is_nonzero(source(row, col));
        }
        result[row] = num_nonzeros;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COUNT_NONZEROS_PER_ROW_KERNEL_SIZE_T);


template <typename ValueType, typename IndexType>
void count_nonzero_blocks_per_row(std::shared_ptr<const ReferenceExecutor> exec,
                                  matrix::view::dense<const ValueType> source,
                                  int bs, IndexType* result)
{
    const auto num_rows = source.size[0];
    const auto num_cols = source.size[1];
    const auto num_block_rows = num_rows / bs;
    const auto num_block_cols = num_cols / bs;
    for (size_type brow = 0; brow < num_block_rows; ++brow) {
        IndexType num_nonzero_blocks{};
        for (size_type bcol = 0; bcol < num_block_cols; ++bcol) {
            bool block_nz = false;
            for (int lrow = 0; lrow < bs; ++lrow) {
                for (int lcol = 0; lcol < bs; ++lcol) {
                    const auto row = lrow + bs * brow;
                    const auto col = lcol + bs * bcol;
                    block_nz = block_nz || is_nonzero(source(row, col));
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
}  // namespace reference
}  // namespace kernels
}  // namespace gko
