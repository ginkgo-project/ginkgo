/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#ifndef GKO_CORE_MATRIX_HYBRID_KERNELS_HPP_
#define GKO_CORE_MATRIX_HYBRID_KERNELS_HPP_


#include <ginkgo/core/matrix/dense.hpp>
#include <ginkgo/core/matrix/hybrid.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_HYBRID_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType)      \
    void convert_to_dense(std::shared_ptr<const DefaultExecutor> exec,        \
                          const matrix::Hybrid<ValueType, IndexType> *source, \
                          matrix::Dense<ValueType> *result)

#define GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL(ValueType, IndexType)      \
    void convert_to_csr(std::shared_ptr<const DefaultExecutor> exec,        \
                        const matrix::Hybrid<ValueType, IndexType> *source, \
                        matrix::Csr<ValueType, IndexType> *result)

#define GKO_DECLARE_HYBRID_COUNT_NONZEROS_KERNEL(ValueType, IndexType)      \
    void count_nonzeros(std::shared_ptr<const DefaultExecutor> exec,        \
                        const matrix::Hybrid<ValueType, IndexType> *source, \
                        size_type *result)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                  \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_HYBRID_CONVERT_TO_DENSE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                 \
    GKO_DECLARE_HYBRID_COUNT_NONZEROS_KERNEL(ValueType, IndexType)


namespace omp {
namespace hybrid {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace hybrid
}  // namespace omp


namespace cuda {
namespace hybrid {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace hybrid
}  // namespace cuda


namespace reference {
namespace hybrid {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace hybrid
}  // namespace reference


namespace hip {
namespace hybrid {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace hybrid
}  // namespace hip


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_MATRIX_HYBRID_KERNELS_HPP_
