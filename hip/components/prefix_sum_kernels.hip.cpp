// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/components/prefix_sum_kernels.hpp"


#include <limits>


#include <thrust/scan.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/name_demangling.hpp>


#include "hip/base/thrust.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace components {


#include "common/cuda_hip/components/prefix_sum_kernels.hpp.inc"


}  // namespace components
}  // namespace hip
}  // namespace kernels
}  // namespace gko
