// SPDX-FileCopyrightText: 2025 The Ginkgo authors
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


#if HIP_VERSION >= 60200000
// HIP has __hip_bfloat16 after ROCM 5.6.0 but enough implementation for us
// (conversion and operation overload) after ROCM 6.2.0 which provides more
// native operations support.
#include <hip/hip_bf16.h>

namespace gko {


using vendor_bf16 = __hip_bfloat16;


}


#else


// HIP has hip_bfloat16 but only the type with the operation fallback to the
// single precision
#include <hip/hip_bfloat16.h>


namespace gko {


using vendor_bf16 = hip_bfloat16;


}


#endif
#endif
#endif  // GKO_COMMON_CUDA_HIP_BASE_BF16_ALIAS_HPP_
