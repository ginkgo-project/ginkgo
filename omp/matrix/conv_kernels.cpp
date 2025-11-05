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
namespace omp {
/*
namespace conv2d {

template <typename ValueType>
void conv2d(std::shared_ptr<const DefaultExecutor> exec,
            const std::vector<const gko::matrix::Dense<ValueType>*>& kernels,
            const gko::matrix::Dense<ValueType>* b,
            std::vector<gko::matrix::Dense<ValueType>*>& outputs)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_CONV2D_KERNEL);

}  // namespace conv2d
*/
namespace conv2dsparse {

template <typename ValueType, typename IndexType>
void conv2dsparse(std::shared_ptr<const OmpExecutor> exec,
                  const gko::matrix::Csr<ValueType, IndexType>* kernel,
                  const gko::matrix::Dense<ValueType>* b,
                  gko::matrix::Dense<ValueType>* x)
{
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONV2DSPARSE_KERNEL);

}  // namespace conv2dsparse

}  // namespace omp
}  // namespace kernels
}  // namespace gko
