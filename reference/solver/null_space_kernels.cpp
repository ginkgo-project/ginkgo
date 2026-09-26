// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/null_space_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The NullSpace namespace.
 *
 */
namespace null_space {


template <typename ValueType>
void compute_scaled_column_sums(std::shared_ptr<const ReferenceExecutor> exec,
                                matrix::view::dense<const ValueType> x,
                                remove_complex<ValueType> scale,
                                matrix::view::dense<ValueType> result,
                                array<char>& tmp)
{
    for (size_type j = 0; j < x.size[1]; ++j) {
        result(0, j) = zero<ValueType>();
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            result(0, j) += x(i, j) * scale;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULL_SPACE_COMPUTE_SCALED_COLUMN_SUMS_KERNEL);


template <typename ValueType>
void remove_constant(std::shared_ptr<const ReferenceExecutor> exec,
                     matrix::view::dense<const ValueType> mean,
                     matrix::view::dense<ValueType> x)
{
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            x(i, j) -= mean(0, j);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULL_SPACE_REMOVE_CONSTANT_KERNEL);


}  // namespace null_space
}  // namespace reference
}  // namespace kernels
}  // namespace gko
