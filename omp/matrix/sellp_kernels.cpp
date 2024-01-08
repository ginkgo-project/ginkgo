// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/sellp_kernels.hpp"


#include <array>


#include <omp.h>


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
                    const matrix::Sellp<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c, OutFn out)
{
    GKO_ASSERT(b->get_size()[1] == num_rhs);
    auto slice_lengths = a->get_const_slice_lengths();
    auto slice_sets = a->get_const_slice_sets();
    auto slice_size = a->get_slice_size();
    auto slice_num = ceildiv(a->get_size()[0] + slice_size - 1, slice_size);
#pragma omp parallel for collapse(2)
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row < a->get_size()[0]) {
                std::array<ValueType, num_rhs> partial_sum;
                partial_sum.fill(zero<ValueType>());
                for (size_type i = 0; i < slice_lengths[slice]; i++) {
                    auto val = a->val_at(row, slice_sets[slice], i);
                    auto col = a->col_at(row, slice_sets[slice], i);
                    if (col != invalid_index<IndexType>()) {
#pragma unroll
                        for (size_type j = 0; j < num_rhs; j++) {
                            partial_sum[j] += val * b->at(col, j);
                        }
                    }
                }
#pragma unroll
                for (size_type j = 0; j < num_rhs; j++) {
                    [&] {
                        c->at(global_row, j) =
                            out(global_row, j, partial_sum[j]);
                    }();
                }
            }
        }
    }
}


template <int block_size, typename ValueType, typename IndexType,
          typename OutFn>
void spmv_blocked(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Sellp<ValueType, IndexType>* a,
                  const matrix::Dense<ValueType>* b,
                  matrix::Dense<ValueType>* c, OutFn out)
{
    auto slice_lengths = a->get_const_slice_lengths();
    auto slice_sets = a->get_const_slice_sets();
    auto slice_size = a->get_slice_size();
    auto slice_num = ceildiv(a->get_size()[0] + slice_size - 1, slice_size);
    const auto num_rhs = b->get_size()[1];
    const auto rounded_rhs = num_rhs / block_size * block_size;
#pragma omp parallel for collapse(2)
    for (size_type slice = 0; slice < slice_num; slice++) {
        for (size_type row = 0; row < slice_size; row++) {
            size_type global_row = slice * slice_size + row;
            if (global_row < a->get_size()[0]) {
                std::array<ValueType, block_size> partial_sum;
                for (size_type rhs_base = 0; rhs_base < rounded_rhs;
                     rhs_base += block_size) {
                    partial_sum.fill(zero<ValueType>());
                    for (size_type i = 0; i < slice_lengths[slice]; i++) {
                        auto val = a->val_at(row, slice_sets[slice], i);
                        auto col = a->col_at(row, slice_sets[slice], i);
                        if (col != invalid_index<IndexType>()) {
#pragma unroll
                            for (size_type j = 0; j < block_size; j++) {
                                partial_sum[j] +=
                                    val * b->at(col, j + rhs_base);
                            }
                        }
                    }
#pragma unroll
                    for (size_type j = 0; j < block_size; j++) {
                        [&] {
                            c->at(global_row, j + rhs_base) =
                                out(global_row, j + rhs_base, partial_sum[j]);
                        }();
                    }
                }
                partial_sum.fill(zero<ValueType>());
                for (size_type i = 0; i < slice_lengths[slice]; i++) {
                    auto val = a->val_at(row, slice_sets[slice], i);
                    auto col = a->col_at(row, slice_sets[slice], i);
                    if (col != invalid_index<IndexType>()) {
                        for (size_type j = rounded_rhs; j < num_rhs; j++) {
                            partial_sum[j - rounded_rhs] += val * b->at(col, j);
                        }
                    }
                }
                for (size_type j = rounded_rhs; j < num_rhs; j++) {
                    [&] {
                        c->at(global_row, j) =
                            out(global_row, j, partial_sum[j - rounded_rhs]);
                    }();
                }
            }
        }
    }
}


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Sellp<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const auto num_rhs = b->get_size()[1];
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
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Sellp<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    const auto num_rhs = b->get_size()[1];
    if (num_rhs <= 0) {
        return;
    }
    const auto alpha_val = alpha->at(0, 0);
    const auto beta_val = beta->at(0, 0);
    auto out = [&](auto i, auto j, auto value) {
        return alpha_val * value + beta_val * c->at(i, j);
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


}  // namespace sellp
}  // namespace omp
}  // namespace kernels
}  // namespace gko
