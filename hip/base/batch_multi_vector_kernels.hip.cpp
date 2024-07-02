// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/base/batch_multi_vector_kernels.hpp"

#include <thrust/functional.h>
#include <thrust/transform.h>

#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/range_accessors.hpp>

#include "common/cuda_hip/base/blas_bindings.hpp"
#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/pointer_mode_guard.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "common/cuda_hip/components/uninitialized_array.hpp"
#include "core/base/batch_struct.hpp"
#include "hip/base/batch_struct.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The MultiVector matrix format namespace.
 *
 * @ingroup batch_multi_vector
 */
namespace batch_multi_vector {


constexpr auto default_block_size = 256;
constexpr int sm_oversubscription = 4;


// clang-format off

// NOTE: DO NOT CHANGE THE ORDERING OF THE INCLUDES

#include "common/cuda_hip/base/batch_multi_vector_kernels.hpp.inc"


#include "common/cuda_hip/base/batch_multi_vector_kernel_launcher.hpp.inc"

// clang-format on


}  // namespace batch_multi_vector
}  // namespace hip
}  // namespace kernels
}  // namespace gko
