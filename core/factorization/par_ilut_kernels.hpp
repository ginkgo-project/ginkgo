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

#ifndef GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_
#define GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_


#include <ginkgo/core/factorization/par_ilut.hpp>


#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


namespace gko {
namespace kernels {


#define GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL(ValueType, IndexType) \
    void add_candidates(std::shared_ptr<const DefaultExecutor> exec,     \
                        const matrix::Csr<ValueType, IndexType> *lu,     \
                        const matrix::Csr<ValueType, IndexType> *a,      \
                        const matrix::Csr<ValueType, IndexType> *l,      \
                        const matrix::Csr<ValueType, IndexType> *u,      \
                        matrix::Csr<ValueType, IndexType> *l_new,        \
                        matrix::Csr<ValueType, IndexType> *u_new)
#define GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL(ValueType, IndexType) \
    void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,    \
                             const matrix::Csr<ValueType, IndexType> *a,     \
                             matrix::Csr<ValueType, IndexType> *l,           \
                             const matrix::Coo<ValueType, IndexType> *l_coo, \
                             matrix::Csr<ValueType, IndexType> *u,           \
                             const matrix::Coo<ValueType, IndexType> *u_coo, \
                             matrix::Csr<ValueType, IndexType> *u_csc)
#define GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL(ValueType, IndexType) \
    void threshold_select(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Csr<ValueType, IndexType> *m,      \
                          IndexType rank, Array<ValueType> &tmp,           \
                          Array<remove_complex<ValueType>> &tmp2,          \
                          remove_complex<ValueType> &threshold)
#define GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL(ValueType, IndexType) \
    void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,     \
                          const matrix::Csr<ValueType, IndexType> *m,      \
                          remove_complex<ValueType> threshold,             \
                          matrix::Csr<ValueType, IndexType> *m_out,        \
                          matrix::Coo<ValueType, IndexType> *m_out_coo,    \
                          bool lower)
#define GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL(ValueType,        \
                                                            IndexType)        \
    void threshold_filter_approx(std::shared_ptr<const DefaultExecutor> exec, \
                                 const matrix::Csr<ValueType, IndexType> *m,  \
                                 IndexType rank, Array<ValueType> &tmp,       \
                                 remove_complex<ValueType> &threshold,        \
                                 matrix::Csr<ValueType, IndexType> *m_out,    \
                                 matrix::Coo<ValueType, IndexType> *m_out_coo)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                      \
    constexpr auto sampleselect_searchtree_height = 8;                    \
    constexpr auto sampleselect_oversampling = 4;                         \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL(ValueType, IndexType);   \
    template <typename ValueType, typename IndexType>                     \
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL(ValueType, IndexType)


namespace omp {
namespace par_ilut_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilut_factorization
}  // namespace omp


namespace cuda {
namespace par_ilut_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilut_factorization
}  // namespace cuda


namespace reference {
namespace par_ilut_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilut_factorization
}  // namespace reference


namespace hip {
namespace par_ilut_factorization {

GKO_DECLARE_ALL_AS_TEMPLATES;

}  // namespace par_ilut_factorization
}  // namespace hip


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_FACTORIZATION_PAR_ILUT_KERNELS_HPP_
