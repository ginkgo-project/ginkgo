/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#ifndef GKO_CORE_PRECONDITIONER_BATCH_ILU_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_ILU_KERNELS_HPP_


#include <ginkgo/core/preconditioner/batch_ilu.hpp>


#include <ginkgo/core/matrix/batch_csr.hpp>


namespace gko {
namespace kernels {


/**
 * @fn generate
 *
 * This kernel builds an ILU preconditioner for each matrix in
 * the input batch of matrices
 *
 * @param exec  The executor on which to run the kernel.
 * @param a  The batch of matrices for which to build the preconditioner.
 */
#define GKO_DECLARE_BATCH_ILU_SPLIT_GENERATE_KERNEL(_type)                    \
    void generate_split(std::shared_ptr<const DefaultExecutor> exec,          \
                        gko::preconditioner::batch_factorization_type f_type, \
                        const matrix::BatchCsr<_type>* a,                     \
                        matrix::BatchCsr<_type>* l,                           \
                        matrix::BatchCsr<_type>* u)

#define GKO_DECLARE_BATCH_ILU_SPLIT_APPLY_KERNEL(_type)                     \
    void apply_split(                                                       \
        std::shared_ptr<const DefaultExecutor> exec,                        \
        const matrix::BatchCsr<_type>* l, const matrix::BatchCsr<_type>* u, \
        const matrix::BatchDense<_type>* r, matrix::BatchDense<_type>* z)


#define GKO_DECLARE_ALL_AS_TEMPLATES                        \
    template <typename ValueType>                           \
    GKO_DECLARE_BATCH_ILU_SPLIT_GENERATE_KERNEL(ValueType); \
    template <typename ValueType>                           \
    GKO_DECLARE_BATCH_ILU_SPLIT_APPLY_KERNEL(ValueType)


namespace omp {
namespace batch_ilu {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ilu
}  // namespace omp


namespace cuda {
namespace batch_ilu {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ilu
}  // namespace cuda


namespace reference {
namespace batch_ilu {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ilu
}  // namespace reference


namespace hip {
namespace batch_ilu {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ilu
}  // namespace hip


namespace dpcpp {
namespace batch_ilu {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace batch_ilu
}  // namespace dpcpp


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
