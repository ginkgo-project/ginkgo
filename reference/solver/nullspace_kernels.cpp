// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/solver/nullspace_kernels.hpp"

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>


namespace gko {
namespace kernels {
namespace reference {
/**
 * @brief The Nullspace projection namespace.
 *
 */
namespace nullspace {


template <typename ValueType>
void compute_coefficients(std::shared_ptr<const ReferenceExecutor> exec,
                          matrix::view::dense<const ValueType> x,
                          matrix::view::dense<const ValueType> basis,
                          bool has_constant, remove_complex<ValueType> inv_size,
                          matrix::view::dense<ValueType> coefficients,
                          array<char>& tmp)
{
    const size_type offset = has_constant ? 1 : 0;
    for (size_type l = 0; l < coefficients.size[0]; ++l) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            coefficients(l, j) = zero<ValueType>();
        }
    }
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            if (has_constant) {
                coefficients(0, j) += x(i, j) * inv_size;
            }
            for (size_type l = 0; l < basis.size[1]; ++l) {
                coefficients(l + offset, j) += conj(basis(i, l)) * x(i, j);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULLSPACE_COMPUTE_COEFFICIENTS_KERNEL);


template <typename ValueType>
void subtract_projection(std::shared_ptr<const ReferenceExecutor> exec,
                         matrix::view::dense<const ValueType> basis,
                         bool has_constant,
                         matrix::view::dense<const ValueType> coefficients,
                         matrix::view::dense<const ValueType> x,
                         matrix::view::dense<ValueType> output)
{
    const size_type offset = has_constant ? 1 : 0;
    for (size_type i = 0; i < x.size[0]; ++i) {
        for (size_type j = 0; j < x.size[1]; ++j) {
            auto value = has_constant ? coefficients(0, j) : zero<ValueType>();
            for (size_type l = 0; l < basis.size[1]; ++l) {
                value += basis(i, l) * coefficients(l + offset, j);
            }
            output(i, j) = x(i, j) - value;
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_NULLSPACE_SUBTRACT_PROJECTION_KERNEL);


}  // namespace nullspace
}  // namespace reference
}  // namespace kernels
}  // namespace gko
