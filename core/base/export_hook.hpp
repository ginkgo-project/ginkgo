// SPDX-FileCopyrightText: 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_EXPORT_HOOK_HPP_
#define GKO_CORE_BASE_EXPORT_HOOK_HPP_


// GKO_EXPORT_HOOK should only be used in the source files for backends.


#ifdef ginkgo_reference_EXPORTS


#include <ginkgo/export_reference.hpp>

#define GKO_EXPORT_HOOK GKO_REFERENCE_EXPORT


#endif


#ifdef ginkgo_omp_EXPORTS


#include <ginkgo/export_omp.hpp>

#define GKO_EXPORT_HOOK GKO_OMP_EXPORT


#endif


#ifdef ginkgo_cuda_EXPORTS


#include <ginkgo/export_cuda.hpp>

#define GKO_EXPORT_HOOK GKO_CUDA_EXPORT


#endif


#ifdef ginkgo_hip_EXPORTS


#include <ginkgo/export_hip.hpp>

#define GKO_EXPORT_HOOK GKO_HIP_EXPORT


#endif


#ifdef ginkgo_dpcpp_EXPORTS


#include <ginkgo/export_dpcpp.hpp>

#define GKO_EXPORT_HOOK GKO_DPCPP_EXPORT


#endif


#endif  // GKO_CORE_BASE_EXPORT_HOOK_HPP_
