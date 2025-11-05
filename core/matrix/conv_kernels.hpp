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
              const array<ValueType>& kernel,              \
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

/*Declarations for conv2d */

#define GKO_DECLARE_CONV2D_KERNEL(ValueType)                              \
    void conv2d(                                                          \
        std::shared_ptr<const DefaultExecutor> exec,                      \
        const std::vector<const gko::matrix::Dense<ValueType>*>& kernels, \
        const gko::matrix::Dense<ValueType>* b,                           \
        std::vector<gko::matrix::Dense<ValueType>*>& outputs)


#define GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D \
    template <typename ValueType>           \
    GKO_DECLARE_CONV2D_KERNEL(ValueType)


namespace omp {
namespace conv2d {

GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D;

}  // namespace conv2d
}  // namespace omp


namespace cuda {
namespace conv2d {

GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D;

}  // namespace conv2d
}  // namespace cuda


namespace reference {
namespace conv2d {

GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D;

}  // namespace conv2d
}  // namespace reference


namespace hip {
namespace conv2d {

GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D;

}  // namespace conv2d
}  // namespace hip


namespace dpcpp {
namespace conv2d {

GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D;

}  // namespace conv2d
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES_CONV2D

/* Declarations for conv2dsparse */
#define GKO_DECLARE_CONV2DSPARSE_KERNEL(ValueType, IndexType)               \
    void conv2dsparse(std::shared_ptr<const DefaultExecutor> exec,          \
                      const gko::matrix::Csr<ValueType, IndexType>* kernel, \
                      const gko::matrix::Dense<ValueType>* b,               \
                      gko::matrix::Dense<ValueType>* x)


#define GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE     \
    template <typename ValueType, typename IndexType> \
    GKO_DECLARE_CONV2DSPARSE_KERNEL(ValueType, IndexType)


namespace omp {
namespace conv2dsparse {

GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE;

}  // namespace conv2dsparse
}  // namespace omp


namespace cuda {
namespace conv2dsparse {
GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE;
}  // namespace conv2dsparse
}  // namespace cuda


namespace reference {
namespace conv2dsparse {
GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE;
}  // namespace conv2dsparse
}  // namespace reference

namespace hip {
namespace conv2dsparse {
GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE;
}  // namespace conv2dsparse
}  // namespace hip

namespace dpcpp {
namespace conv2dsparse {
GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE;
}  // namespace conv2dsparse
}  // namespace dpcpp

#undef GKO_DECLARE_ALL_AS_TEMPLATES_CONV2DSPARSE


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_CONV_KERNELS_HPP_
