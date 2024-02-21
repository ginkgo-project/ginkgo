// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_RANDLIB_BINDINGS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_RANDLIB_BINDINGS_HPP_


#ifdef GKO_COMPILING_HIP
#include "hip/base/hiprand_bindings.hip.hpp"

#define RANDLIB_RNG_PSEUDO_DEFAULT HIPRAND_RNG_PSEUDO_DEFAULT
#else  // GKO_COMPILING_CUDA
#include "cuda/base/curand_bindings.cuh"

#define RANDLIB_RNG_PSEUDO_DEFAULT CURAND_RNG_PSEUDO_DEFAULT
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_RANDLIB_BINDINGS_HPP_
