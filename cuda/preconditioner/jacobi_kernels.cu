// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/runtime.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "core/base/extended_float.hpp"
#include "core/preconditioner/jacobi_utils.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/math.hpp"
#include "cuda/components/thread_ids.cuh"
#include "cuda/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


// a total of 32/16 warps (1024 threads)
#if defined(GKO_COMPILING_HIP) && GINKGO_HIP_PLATFORM_HCC
constexpr int default_num_warps = 16;
#else  // !defined(GKO_COMPILING_HIP) || GINKGO_HIP_PLATFORM_NVCC
constexpr int default_num_warps = 32;
#endif
// with current architectures, at most 32 warps can be scheduled per SM (and
// current GPUs have at most 84 SMs)
constexpr int default_grid_size = 32 * 32 * 128;


#include "common/cuda_hip/preconditioner/jacobi_kernels.hpp.inc"


}  // namespace jacobi
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
