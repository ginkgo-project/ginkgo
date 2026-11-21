// SPDX-FileCopyrightText: 2025 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_DEV_LAPACK_BINDINGS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_DEV_LAPACK_BINDINGS_HPP_


#include <ginkgo/config.hpp>


#if GKO_HAVE_LAPACK


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/cusolver_bindings.hpp"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/hipsolver_bindings.hip.hpp"
#else
#error "Executor definition missing"
#endif


#endif  // GKO_HAVE_LAPACK


#endif  // GKO_COMMON_CUDA_HIP_BASE_DEV_LAPACK_BINDINGS_HPP_
