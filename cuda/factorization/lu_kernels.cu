// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/lu_kernels.hpp"


#include <algorithm>
#include <memory>


#include <thrust/copy.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "cuda/base/thrust.cuh"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/syncfree.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


constexpr static int default_block_size = 512;


#include "common/cuda_hip/factorization/lu_kernels.hpp.inc"


}  // namespace lu_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
