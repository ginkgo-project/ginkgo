// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"

#include <array>

#include <omp.h>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The SELL-P matrix format namespace.
 *
 * @ingroup sellp
 */
namespace sellp {


template <int num_rhs, typename ValueType, typename IndexType, typename OutFn>
void spmv_small_rhs(std::shared_ptr<const OmpExecutor> exec,
                    matrix::view::sellp<const ValueType, const IndexType> a,
                    matrix::view::dense<const ValueType> b,
                    matrix::view::dense<ValueType> c, OutFn out)
{
    GKO_ASSERT(b.size[1] == num_rhs);
    auto slice_lengths = a.slice_lengths;
    auto slice_sets = a.slice_sets;
    auto slice_size = a.slice_size;
    auto slice_num = ceildiv(a.size[0] + slice_size - 1, slice_size);
#pragma omp parallel for collapse(2)
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row < a.size[0]) {
                std::array<ValueType, num_rhs> partial_sum;
                partial_sum.fill(zero<ValueType>());
                for (size_type i = 0; i < slice_lengths[slice]; i++) {
                    auto val = a.val_at(row, slice_sets[slice], i);
                    auto col = a.col_at(row, slice_sets[slice], i);
                    if (col != invalid_index<IndexType>()) {
#pragma unroll
                        for (size_type j = 0; j < num_rhs; j++) {
                            partial_sum[j] += val * b(col, j);
                        }
                    }
                }
#pragma unroll
                for (size_type j = 0; j < num_rhs; j++) {
                    [&] {
                        c(global_row, j) = out(global_row, j, partial_sum[j]);
                    }();
                }
            }
        }
    }
}


template <int block_size, typename ValueType, typename IndexType,
          typename OutFn>
void spmv_blocked(std::shared_ptr<const OmpExecutor> exec,
                  matrix::view::sellp<const ValueType, const IndexType> a,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> c, OutFn out)
{
    auto slice_lengths = a.slice_lengths;
    auto slice_sets = a.slice_sets;
    auto slice_size = a.slice_size;
    auto slice_num = ceildiv(a.size[0] + slice_size - 1, slice_size);
    const auto num_rhs = b.size[1];
    const auto rounded_rhs = num_rhs / block_size * block_size;
#pragma omp parallel for collapse(2)
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row < a.size[0]) {
                std::array<ValueType, block_size> partial_sum;
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
                    partial_sum.fill(zero<ValueType>());
                    for (size_type i = 0; i < slice_lengths[slice]; i++) {
                        auto val = a.val_at(row, slice_sets[slice], i);
                        auto col = a.col_at(row, slice_sets[slice], i);
                        if (col != invalid_index<IndexType>()) {
#pragma unroll
                            for (size_type j = 0; j < block_size; j++) {
                                partial_sum[j] += val * b(col, j + rhs_base);
                            }
                        }
                    }
#pragma unroll
                    for (size_type j = 0; j < block_size; j++) {
                        [&] {
                            c(global_row, j + rhs_base) =
                                out(global_row, j + rhs_base, partial_sum[j]);
                        }();
                    }
                }
                partial_sum.fill(zero<ValueType>());
                for (size_type i = 0; i < slice_lengths[slice]; i++) {
                    auto val = a.val_at(row, slice_sets[slice], i);
                    auto col = a.col_at(row, slice_sets[slice], i);
                    if (col != invalid_index<IndexType>()) {
                        for (size_type j = rounded_rhs; j < num_rhs; j++) {
                            partial_sum[j - rounded_rhs] += val * b(col, j);
                        }
                    }
                }
                for (size_type j = rounded_rhs; j < num_rhs; j++) {
                    [&] {
                        c(global_row, j) =
                            out(global_row, j, partial_sum[j - rounded_rhs]);
                    }();
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          matrix::view::sellp<const ValueType, const IndexType> a,
          matrix::view::dense<const ValueType> b,
          matrix::view::dense<ValueType> c)
{
    const auto num_rhs = b.size[1];
    if (num_rhs <= 0) {
        return;
    }
    auto out = [](auto, auto, auto value) { return value; };
    if (num_rhs == 1) {
        spmv_small_rhs<1>(exec, a, b, c, out);
        return;
    }
    if (num_rhs == 2) {
        spmv_small_rhs<2>(exec, a, b, c, out);
        return;
    }
    if (num_rhs == 3) {
        spmv_small_rhs<3>(exec, a, b, c, out);
        return;
    }
    if (num_rhs == 4) {
        spmv_small_rhs<4>(exec, a, b, c, out);
        return;
    }
    spmv_blocked<4>(exec, a, b, c, out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::sellp<const ValueType, const IndexType> a,
                   matrix::view::dense<const ValueType> b,
                   matrix::view::dense<const ValueType> beta,
                   matrix::view::dense<ValueType> c)
{
    const auto num_rhs = b.size[1];
    if (num_rhs <= 0) {
        return;
    }
    const auto alpha_val = alpha(0, 0);
    const auto beta_val = beta(0, 0);
    auto out = [&](auto i, auto j, auto value) {
        return is_zero(beta_val) ? alpha_val * value
                                 : alpha_val * value + beta_val * c(i, j);
    };
    if (num_rhs == 1) {
        spmv_small_rhs<1>(exec, a, b, c, out);
        return;
    }
    if (num_rhs == 2) {
        spmv_small_rhs<2>(exec, a, b, c, out);
        return;
    }
    if (num_rhs == 3) {
        spmv_small_rhs<3>(exec, a, b, c, out);
        return;
    }
    if (num_rhs == 4) {
        spmv_small_rhs<4>(exec, a, b, c, out);
        return;
    }
    spmv_blocked<4>(exec, a, b, c, out);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMV_KERNEL);


// Gustavson's algorithm: parallel over slices/rows, SIMD over dense columns.
template <typename ValueType, typename IndexType>
void spmm(std::shared_ptr<const OmpExecutor> exec,
          matrix::view::sellp<const ValueType, const IndexType> a,
          matrix::view::dense<const ValueType> b,
          matrix::view::dense<ValueType> c)
{
    const auto slice_lengths = a.slice_lengths;
    const auto slice_sets = a.slice_sets;
    const auto slice_size = a.slice_size;
    const auto num_rows = a.size[0];
    const auto num_cols = c.size[1];
    const auto slice_num = ceildiv(num_rows + slice_size - 1, slice_size);

#pragma omp parallel
    {
        array<ValueType> row_acc{exec, num_cols};
        auto* row_acc_vals = row_acc.get_data();

#pragma omp for collapse(2) schedule(static)
        for (size_type slice = 0; slice < slice_num; ++slice) {
            for (size_type row = 0; row < slice_size; ++row) {
                const auto global_row = slice * slice_size + row;
                if (global_row >= num_rows) {
                    continue;
                }
                const auto slice_begin = slice_sets[slice];
                const auto slice_length = slice_lengths[slice];
                std::fill_n(row_acc_vals, num_cols, zero<ValueType>());
                for (size_type idx = 0; idx < slice_length; ++idx) {
                    const auto val = a.val_at(row, slice_begin, idx);
                    const auto col = a.col_at(row, slice_begin, idx);
                    if (col == invalid_index<IndexType>()) {
                        continue;
                    }
#pragma omp simd
                    for (size_type j = 0; j < num_cols; ++j) {
                        row_acc_vals[j] += val * b(col, j);
                    }
                }
#pragma omp simd
                for (size_type j = 0; j < num_cols; ++j) {
                    c(global_row, j) = row_acc_vals[j];
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_SELLP_SPMM_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmm(std::shared_ptr<const OmpExecutor> exec,
                   matrix::view::dense<const ValueType> alpha,
                   matrix::view::sellp<const ValueType, const IndexType> a,
                   matrix::view::dense<const ValueType> b,
                   matrix::view::dense<const ValueType> beta,
                   matrix::view::dense<ValueType> c)
{
    const auto slice_lengths = a.slice_lengths;
    const auto slice_sets = a.slice_sets;
    const auto slice_size = a.slice_size;
    const auto num_rows = a.size[0];
    const auto num_cols = c.size[1];
    const auto slice_num = ceildiv(num_rows + slice_size - 1, slice_size);
    const auto alpha_val = alpha(0, 0);
    const auto beta_val = beta(0, 0);

#pragma omp parallel
    {
        array<ValueType> row_acc{exec, num_cols};
        auto* row_acc_vals = row_acc.get_data();

#pragma omp for collapse(2) schedule(static)
        for (size_type slice = 0; slice < slice_num; ++slice) {
            for (size_type row = 0; row < slice_size; ++row) {
                const auto global_row = slice * slice_size + row;
                if (global_row >= num_rows) {
                    continue;
                }
                const auto slice_begin = slice_sets[slice];
                const auto slice_length = slice_lengths[slice];
                std::fill_n(row_acc_vals, num_cols, zero<ValueType>());
                for (size_type idx = 0; idx < slice_length; ++idx) {
                    const auto val = a.val_at(row, slice_begin, idx);
                    const auto col = a.col_at(row, slice_begin, idx);
                    if (col == invalid_index<IndexType>()) {
                        continue;
                    }
#pragma omp simd
                    for (size_type j = 0; j < num_cols; ++j) {
                        row_acc_vals[j] += val * b(col, j);
                    }
                }
                if (is_zero(beta_val)) {
#pragma omp simd
                    for (size_type j = 0; j < num_cols; ++j) {
                        c(global_row, j) = alpha_val * row_acc_vals[j];
                    }
                } else {
#pragma omp simd
                    for (size_type j = 0; j < num_cols; ++j) {
                        c(global_row, j) = alpha_val * row_acc_vals[j] +
                                           beta_val * c(global_row, j);
                    }
                }
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SELLP_ADVANCED_SPMM_KERNEL);


}  // namespace sellp
}  // namespace omp
}  // namespace kernels
}  // namespace gko
