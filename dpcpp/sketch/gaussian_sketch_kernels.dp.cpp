// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/sketch/gaussian_sketch_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>


namespace gko {
namespace kernels {
namespace dpcpp {
namespace gaussian_sketch {


template <typename ValueType>
void generate(std::shared_ptr<const DefaultExecutor> exec,
              matrix::view::dense<ValueType> sketch_matrix,
              uint64 seed) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_GAUSSIAN_SKETCH_GENERATE);


}  // namespace gaussian_sketch
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
