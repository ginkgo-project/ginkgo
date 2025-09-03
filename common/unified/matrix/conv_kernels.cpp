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
        [] GKO_KERNEL(auto i, auto j, auto x_values, auto kernel_values,
                      auto stride, auto padding, auto kernel_size, auto b_size,
                      auto b_values) {
            int start = i * stride - padding;
            for (gko::size_type j = 0; j < kernel_size; ++j) {
                if (start + j >= 0 && start + j < b_size[0]) {
                    x_values[i] += kernel_values[j] * b_values[start + j];
                }
            }
        },
        x->get_size()[0], 1, x->get_values(), kernel.get_const_data(), stride,
        padding, kernel.get_size(), b->get_size(), b->get_const_values());
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV_KERNEL);

}  // namespace conv
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_MATRIX_CONV_KERNELS_HPP_
