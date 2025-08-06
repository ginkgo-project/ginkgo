// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_MATRIX_CONV_KERNELS_HPP_
#define GKO_CORE_MATRIX_CONV_KERNELS_HPP_


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/conv.hpp>
#include <ginkgo/core/matrix/dense.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_CONV_KERNEL(ValueType)                 \
    void conv(std::shared_ptr<const DefaultExecutor> exec, \
               const gko::array<ValueType>& kernel,       \
              const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* x)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_CONV_KERNEL(ValueType)


namespace omp {
namespace conv {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace conv
}  // namespace omp


namespace cuda {
namespace conv {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace conv
}  // namespace cuda


namespace reference {
namespace conv {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace conv
}  // namespace reference


namespace hip {
namespace conv {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace conv
}  // namespace hip


namespace dpcpp {
namespace conv {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace conv
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CONV_KERNELS_HPP_
