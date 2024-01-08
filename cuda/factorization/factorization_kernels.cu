// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/factorization_kernels.hpp"


#include <ginkgo/core/base/array.hpp>


#include "core/base/array_access.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/csr_builder.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


constexpr int default_block_size{512};


#include "common/cuda_hip/factorization/factorization_kernels.hpp.inc"


}  // namespace factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
