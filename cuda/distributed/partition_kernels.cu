// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/partition_kernels.hpp"


#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scan.h>
#include <thrust/sort.h>


#include "common/unified/base/kernel_launch.hpp"
#include "core/components/fill_array_kernels.hpp"
#include "cuda/base/thrust.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace partition {


#include "common/cuda_hip/distributed/partition_kernels.hpp.inc"


}  // namespace partition
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
