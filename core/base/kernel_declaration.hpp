// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_BASE_KERNEL_DECLARATION_HPP_
#define GKO_CORE_BASE_KERNEL_DECLARATION_HPP_


// clang-format off
#define GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(_kernel_namespace, ...)      \
                                                                             \
    namespace omp {                                                          \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__;                                                             \
    }                                                                        \
    }                                                                        \
    namespace cuda {                                                         \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__;                                                             \
    }                                                                        \
    }                                                                        \
    namespace reference {                                                    \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__;                                                             \
    }                                                                        \
    }                                                                        \
    namespace hip {                                                          \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__;                                                             \
    }                                                                        \
    }                                                                        \
    namespace dpcpp {                                                        \
    namespace _kernel_namespace {                                            \
    __VA_ARGS__;                                                             \
    }                                                                        \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")
// clang-format on


#endif  // GKO_CORE_BASE_KERNEL_DECLARATION_HPP_
