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

#ifndef GKO_CORE_MATRIX_BATCH_MULTI_VECTOR_KERNELS_HPP_
#define GKO_CORE_MATRIX_BATCH_MULTI_VECTOR_KERNELS_HPP_


#include <ginkgo/core/base/batch_multi_vector.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/diagonal.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL(_type)  \
    void scale(std::shared_ptr<const DefaultExecutor> exec, \
               const BatchMultiVector<_type>* alpha,        \
               BatchMultiVector<_type>* x)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL(_type)  \
    void add_scaled(std::shared_ptr<const DefaultExecutor> exec, \
                    const BatchMultiVector<_type>* alpha,        \
                    const BatchMultiVector<_type>* x,            \
                    BatchMultiVector<_type>* y)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL(_type)  \
    void compute_dot(std::shared_ptr<const DefaultExecutor> exec, \
                     const BatchMultiVector<_type>* x,            \
                     const BatchMultiVector<_type>* y,            \
                     BatchMultiVector<_type>* result)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL(_type)  \
    void compute_norm2(std::shared_ptr<const DefaultExecutor> exec, \
                       const BatchMultiVector<_type>* x,            \
                       BatchMultiVector<remove_complex<_type>>* result)

#define GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL(_type)  \
    void copy(std::shared_ptr<const DefaultExecutor> exec, \
              const BatchMultiVector<_type>* x,            \
              BatchMultiVector<_type>* result)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                \
    template <typename ValueType>                                   \
    GKO_DECLARE_BATCH_MULTI_VECTOR_SCALE_KERNEL(ValueType);         \
    template <typename ValueType>                                   \
    GKO_DECLARE_BATCH_MULTI_VECTOR_ADD_SCALED_KERNEL(ValueType);    \
    template <typename ValueType>                                   \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_DOT_KERNEL(ValueType);   \
    template <typename ValueType>                                   \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COMPUTE_NORM2_KERNEL(ValueType); \
    template <typename ValueType>                                   \
    GKO_DECLARE_BATCH_MULTI_VECTOR_COPY_KERNEL(ValueType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_multi_vector,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_BATCH_MULTI_VECTOR_KERNELS_HPP_
