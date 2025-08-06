// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_SPARSELIB_BLOCK_BINDINGS_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_SPARSELIB_BLOCK_BINDINGS_HPP_


#if defined(GKO_COMPILING_CUDA)
#include "cuda/base/cusparse_block_bindings.hpp"
#elif defined(GKO_COMPILING_HIP)
#include "hip/base/hipsparse_block_bindings.hip.hpp"
#else
#error "Executor definition missing"
#endif


#endif  // GKO_COMMON_CUDA_HIP_BASE_SPARSELIB_BLOCK_BINDINGS_HPP_
