// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/conv_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/allocator.hpp"


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
            const gko::matrix::Dense<ValueType>* kernel,
            const gko::matrix::Dense<ValueType>* b,
            gko::matrix::Dense<ValueType>* x)
{
    // implement convolution here
    GKO_NOT_IMPLEMENTED;
}
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV2D_KERNEL);
}  // namespace conv2d

}  // namespace reference
}  // namespace kernels
}  // namespace gko
