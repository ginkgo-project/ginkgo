// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>
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


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const DefaultExecutor> exec,
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
