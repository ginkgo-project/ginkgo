// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/dense_kernels.hpp"


namespace gko {
namespace kernels {
namespace reference {
namespace dense {


template <typename ValueType>
void simple_apply(std::shared_ptr<const ReferenceExecutor> exec,
                  matrix::view::dense<const ValueType> a,
                  matrix::view::dense<const ValueType> b,
                  matrix::view::dense<ValueType> c)
{
    for (size_type row = 0; row < c.size[0]; ++row) {
        for (size_type col = 0; col < c.size[1]; ++col) {
            c(row, col) = zero<ValueType>();
        }
    }

    for (size_type row = 0; row < c.size[0]; ++row) {
        for (size_type inner = 0; inner < a.size[1]; ++inner) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) += a(row, inner) * b(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_SIMPLE_APPLY_KERNEL);


template <typename ValueType>
void apply(std::shared_ptr<const ReferenceExecutor> exec,
           matrix::view::dense<const ValueType> alpha,
           matrix::view::dense<const ValueType> a,
           matrix::view::dense<const ValueType> b,
           matrix::view::dense<const ValueType> beta,
           matrix::view::dense<ValueType> c)
{
    if (is_nonzero(beta(0, 0))) {
        for (size_type row = 0; row < c.size[0]; ++row) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) *= beta(0, 0);
            }
        }
    } else {
        for (size_type row = 0; row < c.size[0]; ++row) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) = zero<ValueType>();
            }
        }
    }

    for (size_type row = 0; row < c.size[0]; ++row) {
        for (size_type inner = 0; inner < a.size[1]; ++inner) {
            for (size_type col = 0; col < c.size[1]; ++col) {
                c(row, col) += alpha(0, 0) * a(row, inner) * b(inner, col);
            }
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DENSE_APPLY_KERNEL);


}  // namespace dense
}  // namespace reference
}  // namespace kernels
}  // namespace gko
