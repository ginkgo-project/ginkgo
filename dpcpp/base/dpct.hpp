// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_DPCPP_BASE_DPCT_HPP_
#define GKO_DPCPP_BASE_DPCT_HPP_


#include <CL/sycl.hpp>


// This is partial extraction from dpct/dpct.hpp of Intel
#if defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#else
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif


#endif  // GKO_DPCPP_BASE_DPCT_HPP_
