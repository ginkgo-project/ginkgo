// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_BASE_CONFIG_HPP_
#define GKO_COMMON_UNIFIED_BASE_CONFIG_HPP_


#if defined(GKO_COMPILING_CUDA)
#define GKO_DEVICE_NAMESPACE cuda
#elif defined(GKO_COMPILING_HIP)
#define GKO_DEVICE_NAMESPACE hip
#elif defined(GKO_COMPILING_DPCPP)
#define GKO_DEVICE_NAMESPACE dpcpp
#elif defined(GKO_COMPILING_OMP)
#define GKO_DEVICE_NAMESPACE omp
#else
#error "This file should only be used inside Ginkgo device compilation"
#endif


#endif  // GKO_COMMON_UNIFIED_BASE_CONFIG_HPP_
