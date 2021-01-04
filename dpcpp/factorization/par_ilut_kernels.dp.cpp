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

#include "core/factorization/par_ilut_kernels.hpp"


#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/utils.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


template <typename ValueType, typename IndexType>
void threshold_select(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *m,
                      IndexType rank, Array<ValueType> &tmp,
                      Array<remove_complex<ValueType>> &,
                      remove_complex<ValueType> &threshold) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_SELECT_KERNEL);


/**
 * Removes all the elements from the input matrix for which pred is false.
 * Stores the result in m_out and (if non-null) m_out_coo.
 * pred(row, nz) is called for each entry, where nz is the index in
 * values/col_idxs.
 */
template <typename Predicate, typename ValueType, typename IndexType>
void abstract_filter(std::shared_ptr<const DefaultExecutor> exec,
                     const matrix::Csr<ValueType, IndexType> *m,
                     matrix::Csr<ValueType, IndexType> *m_out,
                     matrix::Coo<ValueType, IndexType> *m_out_coo,
                     Predicate pred) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *m,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType> *m_out,
                      matrix::Coo<ValueType, IndexType> *m_out_coo,
                      bool) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


constexpr auto bucket_count = 1 << sampleselect_searchtree_height;
constexpr auto sample_size = bucket_count * sampleselect_oversampling;


template <typename ValueType, typename IndexType>
void threshold_filter_approx(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *m, IndexType rank,
    Array<ValueType> &tmp, remove_complex<ValueType> &threshold,
    matrix::Csr<ValueType, IndexType> *m_out,
    matrix::Coo<ValueType, IndexType> *m_out_coo) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *a,
                         matrix::Csr<ValueType, IndexType> *l,
                         const matrix::Coo<ValueType, IndexType> *,
                         matrix::Csr<ValueType, IndexType> *u,
                         const matrix::Coo<ValueType, IndexType> *,
                         matrix::Csr<ValueType, IndexType> *u_csc)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL);


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *lu,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Csr<ValueType, IndexType> *u,
                    matrix::Csr<ValueType, IndexType> *l_new,
                    matrix::Csr<ValueType, IndexType> *u_new)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
