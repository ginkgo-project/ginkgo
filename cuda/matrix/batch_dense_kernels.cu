// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/matrix/batch_dense_kernels.hpp"


#include <thrust/functional.h>


#include <ginkgo/core/base/batch_multi_vector.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>


#include "core/base/batch_struct.hpp"
#include "core/matrix/batch_struct.hpp"
#include "cuda/base/batch_struct.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/thrust.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Dense matrix format namespace.
 *
 * @ingroup batch_dense
 */
namespace batch_dense {


constexpr auto default_block_size = 256;
constexpr int sm_oversubscription = 4;

// clang-format off

// NOTE: DO NOT CHANGE THE ORDERING OF THE INCLUDES

#include "common/cuda_hip/matrix/batch_dense_kernels.hpp.inc"


#include "common/cuda_hip/matrix/batch_dense_kernel_launcher.hpp.inc"


// clang-format on


}  // namespace batch_dense
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
