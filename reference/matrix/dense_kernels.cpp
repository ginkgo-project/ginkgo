// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"

#include <algorithm>

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
#include "core/base/mixed_precision_types.hpp"
#include "core/components/prefix_sum_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Dense matrix format namespace.
 * @ref Dense
 * @ingroup dense
 */
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


template <typename InValueType, typename OutValueType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::dense<const InValueType> input,
          matrix::view::dense<OutValueType> output)
{
    for (size_type row = 0; row < input.size[0]; ++row) {
        for (size_type col = 0; col < input.size[1]; ++col) {
            output(row, col) = static_cast<OutValueType>(input(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_CONVERSION_OR_COPY(
    GKO_DECLARE_DENSE_COPY_KERNEL);


template <typename ValueType>
void fill(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::dense<ValueType> mat, ValueType value)
{
    for (size_type row = 0; row < mat.size[0]; ++row) {
        for (size_type col = 0; col < mat.size[1]; ++col) {
            mat(row, col) = value;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_FILL_KERNEL);


template <typename ValueType, typename ScalarType>
void scale(std::shared_ptr<const ReferenceExecutor> exec,
           matrix::view::dense<const ScalarType> alpha,
           matrix::view::dense<ValueType> x)
{
    if (alpha.size[1] == 1) {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                if (is_zero(alpha(0, 0))) {
                    x(i, j) = zero<ValueType>();
                } else {
                    x(i, j) *= alpha(0, 0);
                }
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                x(i, j) *= alpha(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(GKO_DECLARE_DENSE_SCALE_KERNEL);


template <typename ValueType, typename ScalarType>
void inv_scale(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::view::dense<const ScalarType> alpha,
               matrix::view::dense<ValueType> x)
{
    if (alpha.size[1] == 1) {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                x(i, j) /= alpha(0, 0);
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                x(i, j) /= alpha(0, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_INV_SCALE_KERNEL);


template <typename ValueType, typename ScalarType>
void add_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ScalarType> alpha,
                matrix::view::dense<const ValueType> x,
                matrix::view::dense<ValueType> y)
{
    if (alpha.size[1] == 1) {
        if (is_nonzero(alpha(0, 0))) {
            for (size_type i = 0; i < x.size[0]; ++i) {
                for (size_type j = 0; j < x.size[1]; ++j) {
                    y(i, j) += alpha(0, 0) * x(i, j);
                }
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                y(i, j) += alpha(0, j) * x(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_ADD_SCALED_KERNEL);


template <typename ValueType, typename ScalarType>
void sub_scaled(std::shared_ptr<const ReferenceExecutor> exec,
                matrix::view::dense<const ScalarType> alpha,
                matrix::view::dense<const ValueType> x,
                matrix::view::dense<ValueType> y)
{
    if (alpha.size[1] == 1) {
        if (is_nonzero(alpha(0, 0))) {
            for (size_type i = 0; i < x.size[0]; ++i) {
                for (size_type j = 0; j < x.size[1]; ++j) {
                    y(i, j) -= alpha(0, 0) * x(i, j);
                }
            }
        }
    } else {
        for (size_type i = 0; i < x.size[0]; ++i) {
            for (size_type j = 0; j < x.size[1]; ++j) {
                y(i, j) -= alpha(0, j) * x(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_SUB_SCALED_KERNEL);


template <typename ValueType>
void add_scaled_diag(std::shared_ptr<const ReferenceExecutor> exec,
                     matrix::view::dense<const ValueType> alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::view::dense<ValueType> y)
{
    const auto diag_values = x->get_const_values();
    if (is_nonzero(alpha(0, 0))) {
        for (size_type i = 0; i < x->get_size()[0]; i++) {
            y(i, i) += alpha(0, 0) * diag_values[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_ADD_SCALED_DIAG_KERNEL);


template <typename ValueType>
void sub_scaled_diag(std::shared_ptr<const ReferenceExecutor> exec,
                     matrix::view::dense<const ValueType> alpha,
                     const matrix::Diagonal<ValueType>* x,
                     matrix::view::dense<ValueType> y)
{
    const auto diag_values = x->get_const_values();
    if (is_nonzero(alpha(0, 0))) {
        for (size_type i = 0; i < x->get_size()[0]; i++) {
            y(i, i) -= alpha(0, 0) * diag_values[i];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SUB_SCALED_DIAG_KERNEL);


template <typename ValueType>
void compute_dot(std::shared_ptr<const ReferenceExecutor> exec,
                 matrix::view::dense<const ValueType> x,
                 matrix::view::dense<const ValueType> y,
                 matrix::view::dense<ValueType> result, array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += x(i, j) * y(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_DOT_KERNEL);


template <typename ValueType>
void compute_dot_dispatch(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> y,
                          matrix::view::dense<ValueType> result,
                          array<char>& tmp)
{
    compute_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_conj_dot(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::view::dense<const ValueType> x,
                      matrix::view::dense<const ValueType> y,
                      matrix::view::dense<ValueType> result, array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += conj(x(i, j)) * y(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_KERNEL);


template <typename ValueType>
void compute_conj_dot_dispatch(std::shared_ptr<const DefaultExecutor> exec,
                               matrix::view::dense<const ValueType> x,
                               matrix::view::dense<const ValueType> y,
                               matrix::view::dense<ValueType> result,
                               array<char>& tmp)
{
    compute_conj_dot(exec, x, y, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_CONJ_DOT_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm2(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::view::dense<const ValueType> x,
                   matrix::view::dense<remove_complex<ValueType>> result,
                   array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<remove_complex<ValueType>>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += squared_norm(x(i, j));
        }
    }
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = sqrt(result(0, j));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM2_KERNEL);


template <typename ValueType>
void compute_norm2_dispatch(
    std::shared_ptr<const DefaultExecutor> exec,
    matrix::view::dense<const ValueType> x,
    matrix::view::dense<remove_complex<ValueType>> result, array<char>& tmp)
{
    compute_norm2(exec, x, result, tmp);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_NORM2_DISPATCH_KERNEL);


template <typename ValueType>
void compute_norm1(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::view::dense<const ValueType> x,
                   matrix::view::dense<remove_complex<ValueType>> result,
                   array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<remove_complex<ValueType>>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += abs(x(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_NORM1_KERNEL);


template <typename ValueType>
void compute_mean(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> x,
                  matrix::view::dense<ValueType> result, array<char>&)
{
    using ValueType_nc = gko::remove_complex<ValueType>;
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }

    if (x.size[0] == 0) return;

    for (size_type i = 0; i < x.size[1]; ++i) {
        for (size_type j = 0; j < x.size[0]; ++j) {
            result(0, i) += x(j, i);
        }
        result(0, i) /= static_cast<ValueType_nc>(x.size[0]);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_COMPUTE_MEAN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const ReferenceExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         matrix::view::dense<ValueType> output)
{
    for (size_type i = 0; i < data.get_num_stored_elements(); i++) {
        output(data.get_const_row_idxs()[i], data.get_const_col_idxs()[i]) =
            data.get_const_values()[i];
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType>
void compute_squared_norm2(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> x,
    matrix::view::dense<remove_complex<ValueType>> result, array<char>&)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<remove_complex<ValueType>>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += squared_norm(x(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_SQUARED_NORM2_KERNEL);


template <typename ValueType>
void compute_sqrt(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<ValueType> data)
{
    for (size_type i = 0; i < data.size[0]; ++i) {
        for (size_type j = 0; j < data.size[1]; ++j) {
            data(i, j) = sqrt(data(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(
    GKO_DECLARE_DENSE_COMPUTE_SQRT_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::dense<const ValueType> source, const int64*,
                    matrix::Coo<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_idxs = result->get_row_idxs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

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
                    matrix::Csr<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto num_nonzeros = result->get_num_stored_elements();

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

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
        for (size_type j = 0; j < result.stride; j++) {
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
                       matrix::Hybrid<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto strategy = result->get_strategy();
    auto ell_lim = strategy->get_ell_num_stored_elements_per_row();
    auto coo_lim = strategy->get_coo_nnz();
    auto coo_val = result->get_coo_values();
    auto coo_col = result->get_coo_col_idxs();
    auto coo_row = result->get_coo_row_idxs();
    for (size_type i = 0; i < result->get_ell_num_stored_elements_per_row();
         i++) {
        for (size_type j = 0; j < result->get_ell_stride(); j++) {
            result->ell_val_at(j, i) = zero<ValueType>();
            result->ell_col_at(j, i) = invalid_index<IndexType>();
        }
    }

    size_type coo_idx = 0;
    for (size_type row = 0; row < num_rows; row++) {
        size_type col = 0;
        for (size_type col_idx = 0; col < num_cols && col_idx < ell_lim;
             col++) {
            auto val = source(row, col);
            if (is_nonzero(val)) {
                result->ell_val_at(row, col_idx) = val;
                result->ell_col_at(row, col_idx) = col;
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
                      matrix::Sellp<ValueType, IndexType>* result)
{
    auto num_rows = result->get_size()[0];
    auto num_cols = result->get_size()[1];
    auto vals = result->get_values();
    auto col_idxs = result->get_col_idxs();
    auto slice_lengths = result->get_slice_lengths();
    auto slice_sets = result->get_slice_sets();
    auto slice_size = result->get_slice_size();
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


template <typename ValueType>
void transpose(std::shared_ptr<const ReferenceExecutor> exec,
               matrix::view::dense<const ValueType> orig,
               matrix::view::dense<ValueType> trans)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            trans(j, i) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_TRANSPOSE_KERNEL);


template <typename ValueType>
void conj_transpose(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::dense<const ValueType> orig,
                    matrix::view::dense<ValueType> trans)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            trans(j, i) = conj(orig(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_CONJ_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void symm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                  const IndexType* perm,
                  matrix::view::dense<const ValueType> orig,
                  matrix::view::dense<ValueType> permuted)
{
    auto size = orig.size[0];
    for (size_type i = 0; i < size; ++i) {
        for (size_type j = 0; j < size; ++j) {
            permuted(i, j) = orig(perm[i], perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                      const IndexType* perm,
                      matrix::view::dense<const ValueType> orig,
                      matrix::view::dense<ValueType> permuted)
{
    auto size = orig.size[0];
    for (size_type i = 0; i < size; ++i) {
        for (size_type j = 0; j < size; ++j) {
            permuted(perm[i], perm[j]) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_SYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void nonsymm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                     const IndexType* row_perm, const IndexType* col_perm,
                     matrix::view::dense<const ValueType> orig,
                     matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            permuted(i, j) = orig(row_perm[i], col_perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_NONSYMM_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_permute(std::shared_ptr<const ReferenceExecutor> exec,
                         const IndexType* row_perm, const IndexType* col_perm,
                         matrix::view::dense<const ValueType> orig,
                         matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            permuted(row_perm[i], col_perm[j]) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_NONSYMM_PERMUTE_KERNEL);


template <typename ValueType, typename OutputType, typename IndexType>
void row_gather(std::shared_ptr<const ReferenceExecutor> exec,
                const IndexType* rows,
                matrix::view::dense<const ValueType> orig,
                matrix::view::dense<OutputType> row_collection)
{
    for (size_type i = 0; i < row_collection.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            row_collection(i, j) = orig(rows[i], j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_DENSE_ROW_GATHER_KERNEL);


template <typename ValueType, typename OutputType, typename IndexType>
void advanced_row_gather(std::shared_ptr<const ReferenceExecutor> exec,
                         matrix::view::dense<const ValueType> alpha,
                         const IndexType* rows,
                         matrix::view::dense<const ValueType> orig,
                         matrix::view::dense<const ValueType> beta,
                         matrix::view::dense<OutputType> row_collection)
{
    using type = highest_precision<ValueType, OutputType>;
    auto scalar_alpha = alpha(0, 0);
    auto scalar_beta = beta(0, 0);
    for (size_type i = 0; i < row_collection.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            row_collection(i, j) =
                static_cast<type>(scalar_alpha * orig(rows[i], j)) +
                static_cast<type>(scalar_beta) *
                    static_cast<type>(row_collection(i, j));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE_2(
    GKO_DECLARE_DENSE_ADVANCED_ROW_GATHER_KERNEL);


template <typename ValueType, typename IndexType>
void col_permute(std::shared_ptr<const ReferenceExecutor> exec,
                 const IndexType* perm,
                 matrix::view::dense<const ValueType> orig,
                 matrix::view::dense<ValueType> col_permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            col_permuted(i, j) = orig(i, perm[j]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COL_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_permute(std::shared_ptr<const ReferenceExecutor> exec,
                     const IndexType* perm,
                     matrix::view::dense<const ValueType> orig,
                     matrix::view::dense<ValueType> row_permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            row_permuted(perm[i], j) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_ROW_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_col_permute(std::shared_ptr<const ReferenceExecutor> exec,
                     const IndexType* perm,
                     matrix::view::dense<const ValueType> orig,
                     matrix::view::dense<ValueType> col_permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            col_permuted(i, perm[j]) = orig(i, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_COL_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void symm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                        const ValueType* scale, const IndexType* perm,
                        matrix::view::dense<const ValueType> orig,
                        matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            const auto col = perm[j];
            permuted(i, j) = scale[row] * scale[col] * orig(row, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_SYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_symm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                            const ValueType* scale, const IndexType* perm,
                            matrix::view::dense<const ValueType> orig,
                            matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            const auto col = perm[j];
            permuted(row, col) = orig(i, j) / (scale[row] * scale[col]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_SYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void nonsymm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                           const ValueType* row_scale,
                           const IndexType* row_perm,
                           const ValueType* col_scale,
                           const IndexType* col_perm,
                           matrix::view::dense<const ValueType> orig,
                           matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = row_perm[i];
            const auto col = col_perm[j];
            permuted(i, j) = row_scale[row] * col_scale[col] * orig(row, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_NONSYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_nonsymm_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                               const ValueType* row_scale,
                               const IndexType* row_perm,
                               const ValueType* col_scale,
                               const IndexType* col_perm,
                               matrix::view::dense<const ValueType> orig,
                               matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = row_perm[i];
            const auto col = col_perm[j];
            permuted(row, col) = orig(i, j) / (row_scale[row] * col_scale[col]);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_NONSYMM_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void row_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       matrix::view::dense<const ValueType> orig,
                       matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            permuted(i, j) = scale[row] * orig(row, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_row_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           matrix::view::dense<const ValueType> orig,
                           matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto row = perm[i];
            permuted(row, j) = orig(i, j) / scale[row];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_ROW_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void col_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                       const ValueType* scale, const IndexType* perm,
                       matrix::view::dense<const ValueType> orig,
                       matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto col = perm[j];
            permuted(i, j) = scale[col] * orig(i, col);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_COL_SCALE_PERMUTE_KERNEL);


template <typename ValueType, typename IndexType>
void inv_col_scale_permute(std::shared_ptr<const ReferenceExecutor> exec,
                           const ValueType* scale, const IndexType* perm,
                           matrix::view::dense<const ValueType> orig,
                           matrix::view::dense<ValueType> permuted)
{
    for (size_type i = 0; i < orig.size[0]; ++i) {
        for (size_type j = 0; j < orig.size[1]; ++j) {
            const auto col = perm[j];
            permuted(i, col) = orig(i, j) / scale[col];
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DENSE_INV_COL_SCALE_PERMUTE_KERNEL);


template <typename ValueType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::view::dense<const ValueType> orig,
                      matrix::Diagonal<ValueType>* diag)
{
    auto diag_values = diag->get_values();
    for (size_type i = 0; i < diag->get_size()[0]; ++i) {
        diag_values[i] = orig(i, i);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType>
void inplace_absolute_dense(std::shared_ptr<const ReferenceExecutor> exec,
                            matrix::view::dense<ValueType> source)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            source(row, col) = abs(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_INPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void outplace_absolute_dense(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::dense<const ValueType> source,
    matrix::view::dense<remove_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = abs(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_OUTPLACE_ABSOLUTE_DENSE_KERNEL);


template <typename ValueType>
void make_complex(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> source,
                  matrix::view::dense<to_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = to_complex<ValueType>{source(row, col)};
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_MAKE_COMPLEX_KERNEL);


template <typename ValueType>
void get_real(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::view::dense<const ValueType> source,
              matrix::view::dense<remove_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = real(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_REAL_KERNEL);


template <typename ValueType>
void get_imag(std::shared_ptr<const ReferenceExecutor> exec,
              matrix::view::dense<const ValueType> source,
              matrix::view::dense<remove_complex<ValueType>> result)
{
    auto dim = source.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            result(row, col) = imag(source(row, col));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GET_IMAG_KERNEL);


template <typename ValueType, typename ScalarType>
void add_scaled_identity(std::shared_ptr<const ReferenceExecutor> exec,
                         matrix::view::dense<const ScalarType> alpha,
                         matrix::view::dense<const ScalarType> beta,
                         matrix::view::dense<ValueType> mtx)
{
    const auto dim = mtx.size;
    for (size_type row = 0; row < dim[0]; row++) {
        for (size_type col = 0; col < dim[1]; col++) {
            mtx(row, col) = beta.values[0] * mtx(row, col);
            if (row == col) {
                mtx(row, row) += alpha.values[0];
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_SCALAR_TYPE(
    GKO_DECLARE_DENSE_ADD_SCALED_IDENTITY_KERNEL);


}  // namespace dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
