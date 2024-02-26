// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/distributed/vector_kernels.hpp"


#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "cuda/base/thrust.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace distributed_vector {


#include "common/cuda_hip/distributed/vector_kernels.hpp.inc"


}  // namespace distributed_vector
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
