// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/ell_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "accessor/reduced_row_major.hpp"
#include "core/base/mixed_precision_types.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The ELL matrix format namespace.
 * @ref Ell
 * @ingroup ell
 */
namespace ell {


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const ReferenceExecutor> exec,
          matrix::view::ell<const MatrixValueType, const IndexType> a,
          matrix::view::dense<const InputValueType> b,
          matrix::view::dense<OutputValueType> c)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    using a_accessor =
        gko::acc::reduced_row_major<1, arithmetic_type, const MatrixValueType,
                                    IndexType>;
    using b_accessor =
        gko::acc::reduced_row_major<2, arithmetic_type, const InputValueType>;

    const auto num_stored_elements_per_row = a.num_stored_elements_per_row;
    const auto stride = a.stride;
    const auto a_vals = gko::acc::range<a_accessor>(
        std::array<acc::size_type, 1>{
            static_cast<acc::size_type>(num_stored_elements_per_row * stride)},
        a.values);
    const auto b_vals = gko::acc::range<b_accessor>(
        std::array<acc::size_type, 2>{{static_cast<acc::size_type>(b.size[0]),
                                       static_cast<acc::size_type>(b.size[1])}},
        b.values,
        std::array<acc::size_type, 1>{{static_cast<acc::size_type>(b.stride)}});

    for (size_type j = 0; j < c.size[1]; j++) {
        for (size_type row = 0; row < a.size[0]; row++) {
            arithmetic_type result{};
            for (size_type i = 0; i < num_stored_elements_per_row; i++) {
                arithmetic_type val = a_vals(row + i * stride);
                auto col = a.col_at(row, i);
                if (col != invalid_index<IndexType>()) {
                    result += val * b_vals(col, j);
                }
            }
            c(row, j) = result;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::view::dense<const MatrixValueType> alpha,
                   matrix::view::ell<const MatrixValueType, const IndexType> a,
                   matrix::view::dense<const InputValueType> b,
                   matrix::view::dense<const OutputValueType> beta,
                   matrix::view::dense<OutputValueType> c)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    using a_accessor =
        gko::acc::reduced_row_major<1, arithmetic_type, const MatrixValueType,
                                    IndexType>;
    using b_accessor =
        gko::acc::reduced_row_major<2, arithmetic_type, const InputValueType>;

    const auto num_stored_elements_per_row = a.num_stored_elements_per_row;
    const auto stride = a.stride;
    const auto a_vals = gko::acc::range<a_accessor>(
        std::array<acc::size_type, 1>{
            static_cast<acc::size_type>(num_stored_elements_per_row * stride)},
        a.values);
    const auto b_vals = gko::acc::range<b_accessor>(
        std::array<acc::size_type, 2>{{static_cast<acc::size_type>(b.size[0]),
                                       static_cast<acc::size_type>(b.size[1])}},
        b.values,
        std::array<acc::size_type, 1>{{static_cast<acc::size_type>(b.stride)}});
    const auto alpha_val = arithmetic_type{alpha(0, 0)};
    const auto beta_val = arithmetic_type{beta(0, 0)};

    for (size_type j = 0; j < c.size[1]; j++) {
        for (size_type row = 0; row < a.size[0]; row++) {
            arithmetic_type result =
                is_zero(beta_val)
                    ? zero<arithmetic_type>()
                    : beta_val * static_cast<arithmetic_type>(c(row, j));
            for (size_type i = 0; i < num_stored_elements_per_row; i++) {
                arithmetic_type val = a_vals(row + i * stride);
                auto col = a.col_at(row, i);
                if (col != invalid_index<IndexType>()) {
                    result += alpha_val * val * b_vals(col, j);
                }
            }
            c(row, j) = result;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


template <typename IndexType>
void compute_max_row_nnz(std::shared_ptr<const DefaultExecutor> exec,
                         const array<IndexType>& row_ptrs, size_type& max_nnz)
{
    max_nnz = 0;
    const auto ptrs = row_ptrs.get_const_data();
    for (size_type i = 1; i < row_ptrs.get_size(); i++) {
        max_nnz = std::max<size_type>(max_nnz, ptrs[i] - ptrs[i - 1]);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ELL_COMPUTE_MAX_ROW_NNZ_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_matrix_data(std::shared_ptr<const DefaultExecutor> exec,
                         const device_matrix_data<ValueType, IndexType>& data,
                         const int64* row_ptrs,
                         matrix::view::ell<ValueType, IndexType> output)
{
    for (size_type row = 0; row < output.size[0]; row++) {
        const auto row_begin = row_ptrs[row];
        const auto row_end = row_ptrs[row + 1];
        size_type col_idx = 0;
        for (auto i = row_begin; i < row_end; i++) {
            output.col_at(row, col_idx) = data.get_const_col_idxs()[i];
            output.val_at(row, col_idx) = data.get_const_values()[i];
            col_idx++;
        }
        for (; col_idx < output.num_stored_elements_per_row; col_idx++) {
            output.col_at(row, col_idx) = invalid_index<IndexType>();
            output.val_at(row, col_idx) = zero<ValueType>();
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_FILL_IN_MATRIX_DATA_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const ReferenceExecutor> exec,
                   matrix::view::ell<const ValueType, const IndexType> source,
                   matrix::view::dense<ValueType> result)
{
    auto num_rows = source.size[0];
    auto num_cols = source.size[1];
    auto num_stored_elements_per_row = source.num_stored_elements_per_row;

    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            const auto col = source.col_at(row, i);
            if (col != invalid_index<IndexType>()) {
                result(row, col) = source.val_at(row, i);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void copy(std::shared_ptr<const DefaultExecutor> exec,
          matrix::view::ell<const ValueType, const IndexType> source,
          matrix::view::ell<ValueType, IndexType> result)
{
    for (size_type row = 0; row < source.size[0]; row++) {
        for (size_type i = 0; i < source.num_stored_elements_per_row; i++) {
            result.col_at(row, i) = source.col_at(row, i);
            result.val_at(row, i) = source.val_at(row, i);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_ELL_COPY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const ReferenceExecutor> exec,
                    matrix::view::ell<const ValueType, const IndexType> source,
                    matrix::Csr<ValueType, IndexType>* result)
{
    const auto num_rows = source.size[0];
    const auto max_nnz_per_row = source.num_stored_elements_per_row;

    auto row_ptrs = result->get_row_ptrs();
    auto col_idxs = result->get_col_idxs();
    auto values = result->get_values();

    size_type cur_ptr = 0;
    row_ptrs[0] = 0;
    for (size_type row = 0; row < num_rows; row++) {
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            const auto val = source.val_at(row, i);
            const auto col = source.col_at(row, i);
            if (col != invalid_index<IndexType>()) {
                values[cur_ptr] = val;
                col_idxs[cur_ptr] = col;
                cur_ptr++;
            }
        }
        row_ptrs[row + 1] = cur_ptr;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros_per_row(
    std::shared_ptr<const ReferenceExecutor> exec,
    matrix::view::ell<const ValueType, const IndexType> source,
    IndexType* result)
{
    const auto num_rows = source.size[0];
    const auto max_nnz_per_row = source.num_stored_elements_per_row;
    const auto stride = source.stride;

    for (size_type row = 0; row < num_rows; row++) {
        size_type nonzeros_in_this_row = 0;
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            if (source.col_at(row, i) != invalid_index<IndexType>()) {
                nonzeros_in_this_row++;
            }
        }
        result[row] = nonzeros_in_this_row;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_COUNT_NONZEROS_PER_ROW_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const ReferenceExecutor> exec,
                      matrix::view::ell<const ValueType, const IndexType> orig,
                      matrix::Diagonal<ValueType>* diag)
{
    const auto col_idxs = orig.col_idxs;
    const auto values = orig.values;
    const auto diag_size = diag->get_size()[0];
    const auto max_nnz_per_row = orig.num_stored_elements_per_row;
    auto diag_values = diag->get_values();

    for (size_type row = 0; row < diag_size; row++) {
        for (size_type i = 0; i < max_nnz_per_row; i++) {
            if (orig.col_at(row, i) == row) {
                diag_values[row] = orig.val_at(row, i);
                break;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_EXTRACT_DIAGONAL_KERNEL);


}  // namespace ell
}  // namespace reference
}  // namespace kernels
}  // namespace gko
