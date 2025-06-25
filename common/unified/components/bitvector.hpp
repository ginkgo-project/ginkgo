// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_UNIFIED_COMPONENTS_BITVECTOR_HPP_
#define GKO_COMMON_UNIFIED_COMPONENTS_BITVECTOR_HPP_


#include "common/unified/base/kernel_launch.hpp"


#if defined(GKO_COMPILING_CUDA) || defined(GKO_COMPILING_HIP)
#include "common/cuda_hip/components/bitvector.hpp"
#elif defined(GKO_COMPILING_OMP)
#include "omp/components/bitvector.hpp"
#elif defined(GKO_COMPILING_DPCPP)
#include "dpcpp/components/bitvector.dp.hpp"
#else
#error "This file should only be used inside Ginkgo device compilation"
#endif


#endif  // GKO_COMMON_UNIFIED_COMPONENTS_BITVECTOR_HPP_
