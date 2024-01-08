// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_PRECONDITIONER_BATCH_PRECONDITIONERS_CUH_
#define GKO_CUDA_PRECONDITIONER_BATCH_PRECONDITIONERS_CUH_


#include <ginkgo/core/matrix/batch_identity.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/reduction.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_preconditioner {


#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"


}  // namespace batch_preconditioner
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_PRECONDITIONER_BATCH_PRECONDITIONERS_CUH_
