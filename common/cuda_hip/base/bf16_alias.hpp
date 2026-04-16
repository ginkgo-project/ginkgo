// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_BF16_ALIAS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_BF16_ALIAS_HPP_


#ifdef GKO_COMPILING_CUDA


#include <cuda_bf16.h>


namespace gko {


using vendor_bf16 = __nv_bfloat16;


}


#elif defined(GKO_COMPILING_HIP)


// HIP has __hip_bfloat16 after ROCM 5.6.0 but enough implementation for us
// (conversion and operation overload) after ROCM 6.2.0 which provides more
// native operations support.
#include <hip/hip_bf16.h>

namespace gko {


using vendor_bf16 = __hip_bfloat16;


}


#endif
#endif  // GKO_COMMON_CUDA_HIP_BASE_BF16_ALIAS_HPP_
