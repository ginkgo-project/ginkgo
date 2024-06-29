// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/index_map_kernels.hpp"

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <thrust/unique.h>

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/components/atomic.hpp"
#include "common/cuda_hip/components/searching.hpp"
#include "cuda/base/thrust.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace index_map {


#include "common/cuda_hip/distributed/index_map_kernels.hpp.inc"


}  // namespace index_map
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
