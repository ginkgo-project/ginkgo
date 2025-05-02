// SPDX-FileCopyrightText: 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_DEV_LAPACK_BINDINGS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_DEV_LAPACK_BINDINGS_HPP_


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/cusolver_bindings.hpp"
#define GKO_DEV_LAPACK_ERROR GKO_CUSOLVER_ERROR
#define DEV_LAPACK_INTERNAL_ERROR CUSOLVER_STATUS_INTERNAL_ERROR
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/hipsolver_bindings.hip.hpp"
#define GKO_DEV_LAPACK_ERROR GKO_HIPSOLVER_ERROR
#define DEV_LAPACK_INTERNAL_ERROR HIPSOLVER_STATUS_INTERNAL_ERROR
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_DEV_LAPACK_BINDINGS_HPP_
