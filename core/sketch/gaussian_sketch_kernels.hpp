// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_SKETCH_GAUSSIAN_SKETCH_KERNELS_HPP_
#define GKO_CORE_SKETCH_GAUSSIAN_SKETCH_KERNELS_HPP_


#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/dense.hpp>

#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_GAUSSIAN_SKETCH_GENERATE(ValueType)            \
    void generate(std::shared_ptr<const DefaultExecutor> exec,     \
                  matrix::view::dense<ValueType> sketch_matrix,    \
                  gko::uint64 seed)


#define GKO_DECLARE_ALL_AS_TEMPLATES \
    template <typename ValueType>    \
    GKO_DECLARE_GAUSSIAN_SKETCH_GENERATE(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(gaussian_sketch,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_SKETCH_GAUSSIAN_SKETCH_KERNELS_HPP_
