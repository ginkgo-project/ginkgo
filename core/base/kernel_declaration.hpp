/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

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
