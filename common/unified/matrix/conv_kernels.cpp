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

}  // namespace conv

namespace conv2d {

template <typename ValueType>
void conv2d(std::shared_ptr<const DefaultExecutor> exec,
            const std::vector<const gko::matrix::Dense<ValueType>*>& kernels,
            const gko::matrix::Dense<ValueType>* b,
            std::vector<gko::matrix::Dense<ValueType>*>& outputs)
{
    int stride_h = 1;
    int stride_w = 1;
    int padding_h = 0;
    int padding_w = 0;

    for (size_t f = 0; f < kernels.size(); ++f) {
        const auto* kernel = kernels[f];
        auto* x = outputs[f];

        run_kernel(
            exec,
            [] GKO_KERNEL(auto i, auto j, auto out_nrows, auto out_ncols,
                          auto kernel_values, auto stride_h, auto stride_w,
                          auto padding_h, auto padding_w, auto kernel_size_h,
                          auto kernel_size_w, auto b, auto x) {
                gko::int64 start_h = i * stride_h - padding_h;
                gko::int64 start_w = j * stride_w - padding_w;
                auto value = zero(kernel_values[0]);
                for (gko::int64 k = 0; k < kernel_size_h; ++k) {
                    if (start_h + k >= 0 && start_h + k < out_nrows) {
                        for (gko::int64 l = 0; l < kernel_size_w; ++l) {
                            if (start_w + l >= 0 && start_w + l < out_ncols) {
                                value += kernel_values[k * kernel_size_w + l] *
                                         b(start_h + k, start_w + l);
                            }
                        }
                    }
                }
                x(i, j) = value;
            },
            x->get_size(), b->get_size()[0], b->get_size()[1],
            kernel->get_const_values(), stride_h, stride_w, padding_h,
            padding_w, kernel->get_size()[0], kernel->get_size()[1], b, x);
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV2D_KERNEL);

// GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE_BASE(GKO_DECLARE_CONV2D_KERNEL);
}  // namespace conv2d
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_MATRIX_CONV_KERNELS_HPP_
