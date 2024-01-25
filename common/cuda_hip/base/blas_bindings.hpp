// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_BLAS_BINDINGS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_BLAS_BINDINGS_HPP_


#include "common/unified/base/config.hpp"


#ifdef GKO_COMPILING_HIP
#include "hip/base/hipblas_bindings.hip.hpp"

#define BLAS_OP_N HIPBLAS_OP_N
#define BLAS_OP_T HIPBLAS_OP_T
#define BLAS_OP_C HIPBLAS_OP_C
#else  // GKO_COMPILING_CUDA
#include "cuda/base/cublas_bindings.cuh"

#define BLAS_OP_N CUBLAS_OP_N
#define BLAS_OP_T CUBLAS_OP_T
#define BLAS_OP_C CUBLAS_OP_C
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_BLAS_BINDINGS_HPP_
