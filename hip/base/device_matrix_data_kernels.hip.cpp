// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/device_matrix_data_kernels.hpp"


#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>


#include "hip/base/thrust.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace components {


#include "common/cuda_hip/base/device_matrix_data_kernels.hpp.inc"


}  // namespace components
}  // namespace hip
}  // namespace kernels
}  // namespace gko
