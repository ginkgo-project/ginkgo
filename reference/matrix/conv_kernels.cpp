// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/conv_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/allocator.hpp"
#include "core/matrix/csr_accessor_helper.hpp"


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Conv matrix format namespace.
 * @ref Conv
 * @ingroup conv
 */
namespace conv {


template <typename ValueType>
void conv(std::shared_ptr<const DefaultExecutor> exec,
          const array<ValueType>& kernel, const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* x)
{
    const auto b_size = b->get_size();                 // (N, 1)
    const auto x_size = x->get_size();                 // (N + K - 1, 1)
    const auto kernel_size = kernel.get_size();        // K
    const auto* kernel_ptr = kernel.get_const_data();  // pointer to kernel data
    int stride = 1;
    int padding = 0;
    // int output_length = (b_size[0] + 2 * padding - kernel_size) / stride + 1;

    for (gko::size_type i = 0; i < x_size[0]; ++i) {
        ValueType sum = zero<ValueType>();
        gko::int64 start = static_cast<gko::int64>(i * stride) - padding;
        for (gko::size_type j = 0; j < kernel_size; ++j) {
            gko::int64 b_idx =
                start +
                static_cast<gko::int64>(
                    j);  // calculate the index in b's row based on the current
                         // position in x and the kernel's stride and padding
            if (b_idx >= 0 && b_idx < static_cast<gko::int64>(b_size[0])) {
                sum += kernel_ptr[j] * b->at(static_cast<gko::size_type>(b_idx),
                                             0);  // direct pointer access
            }
        }
        x->at(i, 0) = sum;
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV_KERNEL);


}  // namespace conv

namespace conv2d {

template <typename ValueType>
void conv2d(std::shared_ptr<const DefaultExecutor> exec,
            const std::vector<const gko::matrix::Dense<ValueType>*>& kernels,
            const gko::matrix::Dense<ValueType>* b,
            std::vector<gko::matrix::Dense<ValueType>*>& outputs)
{
    const auto b_size_row = b->get_size()[0];
    const auto b_size_col = b->get_size()[1];
    const int stride_row = 1;
    const int stride_col = 1;
    const int padding_row = 0;
    const int padding_col = 0;

    for (std::size_t f = 0; f < kernels.size(); ++f) {
        const auto* kernel = kernels[f];
        auto* x = outputs[f];

        const auto x_size_row = x->get_size()[0];
        const auto x_size_col = x->get_size()[1];
        const auto kernel_size_row = kernel->get_size()[0];
        const auto kernel_size_col = kernel->get_size()[1];

        for (gko::size_type i = 0; i < x_size_row; ++i) {
            for (gko::size_type j = 0; j < x_size_col; ++j) {
                ValueType sum = zero<ValueType>();

                gko::int64 start_row =
                    static_cast<gko::int64>(i * stride_row) - padding_row;
                gko::int64 start_col =
                    static_cast<gko::int64>(j * stride_col) - padding_col;

                for (gko::size_type k = 0; k < kernel_size_row; ++k) {
                    gko::int64 b_idx_row =
                        start_row + static_cast<gko::int64>(k);
                    if (b_idx_row >= 0 &&
                        b_idx_row < static_cast<gko::int64>(b_size_row)) {
                        for (gko::size_type l = 0; l < kernel_size_col; ++l) {
                            gko::int64 b_idx_col =
                                start_col + static_cast<gko::int64>(l);
                            if (b_idx_col >= 0 &&
                                b_idx_col <
                                    static_cast<gko::int64>(b_size_col)) {
                                sum += kernel->at(k, l) *
                                       b->at(b_idx_row, b_idx_col);
                            }
                        }
                    }
                }
                x->at(i, j) = sum;
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV2D_KERNEL);
// GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(GKO_DECLARE_CONV2D_KERNEL);

}  // namespace conv2d

namespace conv2dsparse {
// implement convolution here
template <typename ValueType, typename IndexType>
void conv2dsparse(
    std::shared_ptr<const ReferenceExecutor> exec,
    const std::vector<const gko::matrix::Csr<ValueType, IndexType>*>& kernels,
    const gko::matrix::Dense<ValueType>* b,
    std::vector<gko::matrix::Dense<ValueType>*>& x)
{
    // implement convolution here
    /*
    const int b_size_row = b->get_size()[0];
    const int b_size_col = b->get_size()[1];
    const int x_size_row = x->get_size()[0];
    const int x_size_col = x->get_size()[1];
    const int kernel_size_row = kernel->get_size()[0];
    const int kernel_size_col = kernel->get_size()[1];
    int stride_row = 1;
    int stride_col = 1;
    int padding_row = 0;
    int padding_col = 0;
    // int output_size_col = (b_size_col + 2 * padding_col - kernel_size_col) /
    // stride_col + 1;


    // int output_size_row = (b_size_row + 2 * padding_row - kernel_size_row) /
    // stride_row + 1;

    auto row_ptrs = kernel->get_const_row_ptrs();
    auto col_idxs = kernel->get_const_col_idxs();
    // using arithmetic_type = kernel->get_value_type();
    const auto kernel_vals =
        acc::helper::build_const_rrm_accessor<ValueType>(kernel);

    // convolution loop
    for (gko::size_type i = 0; i < x_size_row; ++i) {
        for (gko::size_type j = 0; j < x_size_col; ++j) {
            ValueType sum = zero<ValueType>();
            gko::int64 start_row =
                static_cast<gko::int64>(i * stride_row) - padding_row;
            gko::int64 start_col =
                static_cast<gko::int64>(j * stride_col) - padding_col;
            for (gko::size_type k = 0; k < kernel_vals.length(0); ++k) {
                gko::int64 b_idx_row =
                    start_row + static_cast<gko::int64>(row_ptrs[k]);
                if (b_idx_row >= 0 &&
                    b_idx_row < static_cast<gko::int64>(b_size_row)) {
                    gko::int64 b_idx_col = static_cast<gko::int64>(col_idxs[k]);
                    if (b_idx_col >= 0 &&
                        b_idx_col < static_cast<gko::int64>(b_size_col)) {
                        sum += kernel_vals(k) * b->at(b_idx_row, b_idx_col);
                    }
                }
            }
            x->at(i, j) = sum;
        }
    }
    */
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONV2DSPARSE_KERNEL);

}  // namespace conv2dsparse


}  // namespace reference
}  // namespace kernels
}  // namespace gko
