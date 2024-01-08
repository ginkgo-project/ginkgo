// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/ell_kernels.hpp"


#include <array>


#include <omp.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "accessor/reduced_row_major.hpp"
#include "core/base/mixed_precision_types.hpp"


namespace gko {
namespace kernels {
namespace omp {
/**
 * @brief The ELL matrix format namespace.
 *
 * @ingroup ell
 */
namespace ell {


template <int num_rhs, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType, typename OutFn>
void spmv_small_rhs(std::shared_ptr<const OmpExecutor> exec,
                    const matrix::Ell<MatrixValueType, IndexType>* a,
                    const matrix::Dense<InputValueType>* b,
                    matrix::Dense<OutputValueType>* c, OutFn out)
{
    GKO_ASSERT(b->get_size()[1] == num_rhs);
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    using a_accessor =
        gko::acc::reduced_row_major<1, arithmetic_type, const MatrixValueType>;
    using b_accessor =
        gko::acc::reduced_row_major<2, arithmetic_type, const InputValueType>;

    const auto num_stored_elements_per_row =
        a->get_num_stored_elements_per_row();
    const auto stride = a->get_stride();
    const auto a_vals = gko::acc::range<a_accessor>(
        std::array<acc::size_type, 1>{
            static_cast<acc::size_type>(num_stored_elements_per_row * stride)},
        a->get_const_values());
    const auto b_vals = gko::acc::range<b_accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(b->get_size()[0]),
             static_cast<acc::size_type>(b->get_size()[1])}},
        b->get_const_values(),
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(b->get_stride())}});

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; row++) {
        std::array<arithmetic_type, num_rhs> partial_sum;
        partial_sum.fill(zero<arithmetic_type>());
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            arithmetic_type val = a_vals(row + i * stride);
            auto col = a->col_at(row, i);
            if (col != invalid_index<IndexType>()) {
#pragma unroll
                for (size_type j = 0; j < num_rhs; j++) {
                    partial_sum[j] += val * b_vals(col, j);
                }
            }
        }
#pragma unroll
        for (size_type j = 0; j < num_rhs; j++) {
            [&] { c->at(row, j) = out(row, j, partial_sum[j]); }();
        }
    }
}


template <int block_size, typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType, typename OutFn>
void spmv_blocked(std::shared_ptr<const OmpExecutor> exec,
                  const matrix::Ell<MatrixValueType, IndexType>* a,
                  const matrix::Dense<InputValueType>* b,
                  matrix::Dense<OutputValueType>* c, OutFn out)
{
    GKO_ASSERT(b->get_size()[1] > block_size);
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    using a_accessor =
        gko::acc::reduced_row_major<1, arithmetic_type, const MatrixValueType>;
    using b_accessor =
        gko::acc::reduced_row_major<2, arithmetic_type, const InputValueType>;

    const auto num_stored_elements_per_row =
        a->get_num_stored_elements_per_row();
    const auto stride = a->get_stride();
    const auto a_vals = gko::acc::range<a_accessor>(
        std::array<acc::size_type, 1>{
            static_cast<acc::size_type>(num_stored_elements_per_row * stride)},
        a->get_const_values());
    const auto b_vals = gko::acc::range<b_accessor>(
        std::array<acc::size_type, 2>{
            {static_cast<acc::size_type>(b->get_size()[0]),
             static_cast<acc::size_type>(b->get_size()[1])}},
        b->get_const_values(),
        std::array<acc::size_type, 1>{
            {static_cast<acc::size_type>(b->get_stride())}});

    const auto num_rhs = b->get_size()[1];
    const auto rounded_rhs = num_rhs / block_size * block_size;

#pragma omp parallel for
    for (size_type row = 0; row < a->get_size()[0]; row++) {
        std::array<arithmetic_type, block_size> partial_sum;
        for (size_type rhs_base = 0; rhs_base < rounded_rhs;
             rhs_base += block_size) {
            partial_sum.fill(zero<arithmetic_type>());
            for (size_type i = 0; i < num_stored_elements_per_row; i++) {
                arithmetic_type val = a_vals(row + i * stride);
                auto col = a->col_at(row, i);
                if (col != invalid_index<IndexType>()) {
#pragma unroll
                    for (size_type j = 0; j < block_size; j++) {
                        partial_sum[j] += val * b_vals(col, j + rhs_base);
                    }
                }
            }
#pragma unroll
            for (size_type j = 0; j < block_size; j++) {
                const auto col = j + rhs_base;
                [&] { c->at(row, col) = out(row, col, partial_sum[j]); }();
            }
        }
        partial_sum.fill(zero<arithmetic_type>());
        for (size_type i = 0; i < num_stored_elements_per_row; i++) {
            arithmetic_type val = a_vals(row + i * stride);
            auto col = a->col_at(row, i);
            if (col != invalid_index<IndexType>()) {
                for (size_type j = rounded_rhs; j < num_rhs; j++) {
                    partial_sum[j - rounded_rhs] += val * b_vals(col, j);
                }
            }
        }
        for (size_type j = rounded_rhs; j < num_rhs; j++) {
            [&] {
                c->at(row, j) = out(row, j, partial_sum[j - rounded_rhs]);
            }();
        }
    }
}


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void spmv(std::shared_ptr<const OmpExecutor> exec,
          const matrix::Ell<MatrixValueType, IndexType>* a,
          const matrix::Dense<InputValueType>* b,
          matrix::Dense<OutputValueType>* c)
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

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_SPMV_KERNEL);


template <typename InputValueType, typename MatrixValueType,
          typename OutputValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const OmpExecutor> exec,
                   const matrix::Dense<MatrixValueType>* alpha,
                   const matrix::Ell<MatrixValueType, IndexType>* a,
                   const matrix::Dense<InputValueType>* b,
                   const matrix::Dense<OutputValueType>* beta,
                   matrix::Dense<OutputValueType>* c)
{
    using arithmetic_type =
        highest_precision<InputValueType, OutputValueType, MatrixValueType>;
    const auto num_rhs = b->get_size()[1];
    if (num_rhs <= 0) {
        return;
    }
    const auto alpha_val = arithmetic_type{alpha->at(0, 0)};
    const auto beta_val = arithmetic_type{beta->at(0, 0)};
    auto out = [&](auto i, auto j, auto value) {
        return alpha_val * value + beta_val * arithmetic_type{c->at(i, j)};
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

GKO_INSTANTIATE_FOR_EACH_MIXED_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ELL_ADVANCED_SPMV_KERNEL);


}  // namespace ell
}  // namespace omp
}  // namespace kernels
}  // namespace gko
