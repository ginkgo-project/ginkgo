// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_MATRIX_CONV_KERNELS_HPP_
#define GKO_COMMON_UNIFIED_MATRIX_CONV_KERNELS_HPP_

#include "core/matrix/conv_kernels.hpp"

#include <memory>

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "common/unified/base/kernel_launch.hpp"
#include "common/unified/base/kernel_launch_reduction.hpp"

namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
namespace conv {


template <typename ValueType>
void conv(std::shared_ptr<const DefaultExecutor> exec,
          const array<ValueType>& kernel, const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* x)
{
    int stride = 1;
    int padding = 0;
    run_kernel(
        exec,
        [] GKO_KERNEL(auto i, auto out_nrows, auto kernel_values, auto stride,
                      auto padding, auto kernel_size, auto b, auto x) {
            gko::int64 start = i * stride - padding;
            auto value = zero(kernel_values[0]);
            for (gko::int64 j = 0; j < kernel_size; ++j) {
                if (start + j >= 0 && start + j < out_nrows) {
                    value += kernel_values[j] * b(start + j, 0);
                }
            }
            x(i, 0) = value;
        },
        x->get_size()[0], b->get_size()[0], kernel.get_const_data(), stride,
        padding, kernel.get_size(), b, x);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV_KERNEL);
}

// =============================================================
// 2D convolution kernel
// =============================================================

namespace conv2d {
template <typename ValueType>
void conv2d(std::shared_ptr<const DefaultExecutor> exec,
            const std::vector<const gko::matrix::Dense<ValueType>*>& kernels,
            const gko::matrix::Dense<ValueType>* b,
            std::vector<gko::matrix::Dense<ValueType>*>& outputs)
{
    using gko::zero;

    int stride_h = 1;
    int stride_w = 1;
    int padding_h = 0;
    int padding_w = 0;

    auto in_h = b->get_size()[0];
    auto in_w = b->get_size()[1];

    for (size_t f = 0; f < kernels.size(); ++f) {
        const auto* kernel = kernels[f];
        auto* x = outputs[f];

        auto kernel_h = kernel->get_size()[0];
        auto kernel_w = kernel->get_size()[1];
        auto out_h = x->get_size()[0];
        auto out_w = x->get_size()[1];

        const auto* kernel_values = kernel->get_const_values();

        run_kernel(
            exec,
            [] GKO_KERNEL(auto row, auto col, auto in_h, auto in_w,
                          auto kernel_values, auto stride_h, auto stride_w,
                          auto padding_h, auto padding_w, auto kernel_h,
                          auto kernel_w, auto b, auto x) {
                using value_type = std::decay_t<decltype(x(row, col))>;
                value_type value = gko::zero<value_type>();

                for (gko::int64 kh = 0; kh < kernel_h; ++kh) {
                    for (gko::int64 kw = 0; kw < kernel_w; ++kw) {
                        const gko::int64 ih = row * stride_h - padding_h + kh;
                        const gko::int64 iw = col * stride_w - padding_w + kw;

                        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                            value += static_cast<value_type>(
                                kernel_values[kh * kernel_w + kw]) *
                                static_cast<value_type>(b(ih, iw));
                        }
                    }
                }
                x(row, col) = value;
            },
            out_h, out_w,
            in_h, in_w, kernel_values,
            stride_h, stride_w, padding_h, padding_w,
            kernel_h, kernel_w,
            b, x);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV2D_KERNEL);


}  // namespace conv
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko

#endif  // GKO_COMMON_UNIFIED_MATRIX_CONV_KERNELS_HPP_

