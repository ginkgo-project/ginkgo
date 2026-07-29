// SPDX-FileCopyrightText: 2017 - 2026 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_KERNEL_DECLARATION_HPP_
#define GKO_CORE_BASE_KERNEL_DECLARATION_HPP_

#include <ginkgo/core/base/export_cuda.hpp>
#include <ginkgo/core/base/export_dpcpp.hpp>
#include <ginkgo/core/base/export_hip.hpp>
#include <ginkgo/core/base/export_omp.hpp>
#include <ginkgo/core/base/export_reference.hpp>

// clang-format off
#define GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(_kernel_namespace, ...)      \
                                                                             \
    namespace omp {                                                          \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__(GKO_OMP_EXPORT);                                             \
    }                                                                        \
    }                                                                        \
    namespace cuda {                                                         \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__(GKO_CUDA_EXPORT);                                            \
    }                                                                        \
    }                                                                        \
    namespace reference {                                                    \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__(GKO_REFERENCE_EXPORT);                                       \
    }                                                                        \
    }                                                                        \
    namespace hip {                                                          \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__(GKO_HIP_EXPORT);                                             \
    }                                                                        \
    }                                                                        \
    namespace dpcpp {                                                        \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__(GKO_DPCPP_EXPORT);                                           \
    }                                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
// clang-format on


#endif  // GKO_CORE_BASE_KERNEL_DECLARATION_HPP_
