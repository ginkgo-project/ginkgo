// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_helpers_kernels.hpp"


#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>


#include "cuda/base/thrust.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace partition_helpers {


#include "common/cuda_hip/distributed/partition_helpers_kernels.hpp.inc"


}  // namespace partition_helpers
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
