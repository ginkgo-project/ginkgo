// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_HIP_PRECONDITIONER_BATCH_PRECONDITIONERS_HIP_HPP_
#define GKO_HIP_PRECONDITIONER_BATCH_PRECONDITIONERS_HIP_HPP_


#include <ginkgo/core/matrix/batch_identity.hpp>


#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/reduction.hpp"
#include "core/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace batch_preconditioner {


#include "common/cuda_hip/preconditioner/batch_identity.hpp.inc"


}  // namespace batch_preconditioner
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_PRECONDITIONER_BATCH_PRECONDITIONERS_HIP_HPP_
