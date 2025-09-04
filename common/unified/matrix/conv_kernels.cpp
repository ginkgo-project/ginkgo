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
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko


#endif  // GKO_COMMON_UNIFIED_MATRIX_CONV_KERNELS_HPP_
