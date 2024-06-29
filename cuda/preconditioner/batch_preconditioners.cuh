// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_PRECONDITIONER_BATCH_PRECONDITIONERS_CUH_
#define GKO_CUDA_PRECONDITIONER_BATCH_PRECONDITIONERS_CUH_


#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/preconditioner/batch_jacobi_helpers.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_preconditioner {


#include "common/cuda_hip/preconditioner/batch_block_jacobi.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_scalar_jacobi.hpp.inc"


}  // namespace batch_preconditioner
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_PRECONDITIONER_BATCH_PRECONDITIONERS_CUH_
