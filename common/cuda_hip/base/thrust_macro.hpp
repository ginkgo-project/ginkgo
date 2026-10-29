// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_COMMON_CUDA_HIP_BASE_THRUST_MACRO_HPP_
#define GKO_COMMON_CUDA_HIP_BASE_THRUST_MACRO_HPP_

// although thrust provides the similar thing, these macro are only defined when
// they supported. Thus, we need to provide our own macro to make it work with
// the old version
#ifdef THRUST_CUB_WRAPPED_NAMESPACE
#define GKO_THRUST_NAMESPACE_PREFIX namespace THRUST_CUB_WRAPPED_NAMESPACE {
#define GKO_THRUST_NAMESPACE_POSTFIX }
#define GKO_THRUST_QUALIFIER ::THRUST_CUB_WRAPPED_NAMESPACE::thrust
#else
#define GKO_THRUST_NAMESPACE_PREFIX
#define GKO_THRUST_NAMESPACE_POSTFIX
#define GKO_THRUST_QUALIFIER ::thrust
#endif  // THRUST_CUB_WRAPPED_NAMESPACE


#endif  // GKO_COMMON_CUDA_HIP_BASE_THRUST_MACRO_HPP_
