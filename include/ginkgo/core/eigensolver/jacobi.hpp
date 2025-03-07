// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace experimental {
namespace eigensolver {

template <typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>> jacobi(
    std::shared_ptr<gko::matrix::Dense<ValueType>> matrix, int block_size,
    double tol, int max_iter);

}  // namespace eigensolver
}  // namespace experimental
}  // namespace gko
