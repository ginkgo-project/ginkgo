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

#include "core/factorization/par_ict_kernels.hpp"


#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <unordered_set>


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/base/utils.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ICT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ict_factorization {


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *a,
                    matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Coo<ValueType, IndexType> *)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *llh,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    matrix::Csr<ValueType, IndexType> *l_new)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ict_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
