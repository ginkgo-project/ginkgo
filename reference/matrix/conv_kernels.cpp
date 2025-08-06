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
          const array<ValueType>& kernel,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)
{
    const auto b_size = b->get_size();                      // (N, 1)
    const auto x_size = x->get_size();                      // (N + K - 1, 1)
    const auto kernel_size = kernel.get_size();  // K
    const auto* kernel_ptr = kernel.get_const_data();       // pointer to kernel data

    for (int i = 0; i < x_size[0]; ++i) {
        ValueType sum = zero<ValueType>();
        for (int j = 0; j < kernel_size; ++j) {
            int b_idx = i - j;
            if (b_idx >= 0 && b_idx < b_size[0]) {
                sum += kernel_ptr[j] * b->at(b_idx, 0);  // direct pointer access
            }
        }
        x->at(i, 0) = sum;
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV_KERNEL);


}  // namespace conv
}  // namespace reference
}  // namespace kernels
}  // namespace gko
