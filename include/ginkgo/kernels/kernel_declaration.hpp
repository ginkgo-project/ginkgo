/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#ifndef GKO_PUBLIC_KERNELS_KERNEL_DECLARATION_HPP_
#define GKO_PUBLIC_KERNELS_KERNEL_DECLARATION_HPP_


#include <ginkgo/core/base/executor.hpp>


#define GKO_DECLARE_UNIFIED(...)                                             \
                                                                             \
    namespace omp {                                                          \
    using DefaultExecutor = ::gko::OmpExecutor;                              \
    __VA_ARGS__;                                                             \
    }                                                                        \
    namespace cuda {                                                         \
    using DefaultExecutor = ::gko::CudaExecutor;                             \
    __VA_ARGS__;                                                             \
    }                                                                        \
    namespace reference {                                                    \
    using DefaultExecutor = ::gko::ReferenceExecutor;                        \
    __VA_ARGS__;                                                             \
    }                                                                        \
    namespace hip {                                                          \
    using DefaultExecutor = ::gko::HipExecutor;                              \
    __VA_ARGS__;                                                             \
    }                                                                        \
    namespace dpcpp {                                                        \
    using DefaultExecutor = ::gko::DpcppExecutor;                            \
    __VA_ARGS__;                                                             \
    }                                                                        \
    static_assert(true,                                                      \
                  "This assert is used to counter the false positive extra " \
                  "semi-colon warnings")


#define GKO_REGISTER_UNIFIED_OPERATION(_name, _kernel)                         \
    template <typename... Args>                                                \
    auto make_##_name(Args&&... args)                                          \
    {                                                                          \
        return ::gko::detail::make_register_operation(                         \
            #_name, sizeof...(Args), [&args...](auto exec) {                   \
                using exec_type = decltype(exec);                              \
                if (std::is_same<                                              \
                        exec_type,                                             \
                        std::shared_ptr<const ::gko::ReferenceExecutor>>::     \
                        value) {                                               \
                    reference::_kernel(                                        \
                        std::dynamic_pointer_cast<                             \
                            const ::gko::ReferenceExecutor>(exec),             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::OmpExecutor>>::    \
                               value) {                                        \
                    omp::_kernel(                                              \
                        std::dynamic_pointer_cast<const ::gko::OmpExecutor>(   \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::CudaExecutor>>::   \
                               value) {                                        \
                    cuda::_kernel(                                             \
                        std::dynamic_pointer_cast<const ::gko::CudaExecutor>(  \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::HipExecutor>>::    \
                               value) {                                        \
                    hip::_kernel(                                              \
                        std::dynamic_pointer_cast<const ::gko::HipExecutor>(   \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else if (std::is_same<                                       \
                               exec_type,                                      \
                               std::shared_ptr<const ::gko::DpcppExecutor>>::  \
                               value) {                                        \
                    dpcpp::_kernel(                                            \
                        std::dynamic_pointer_cast<const ::gko::DpcppExecutor>( \
                            exec),                                             \
                        std::forward<Args>(args)...);                          \
                } else {                                                       \
                    GKO_NOT_IMPLEMENTED;                                       \
                }                                                              \
            });                                                                \
    }                                                                          \
    static_assert(true,                                                        \
                  "This assert is used to counter the false positive extra "   \
                  "semi-colon warnings")


#endif  // GKO_PUBLIC_KERNELS_KERNEL_DECLARATION_HPP_
