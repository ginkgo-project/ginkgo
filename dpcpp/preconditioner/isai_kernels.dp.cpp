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

#include "core/preconditioner/isai_kernels.hpp"


#include <algorithm>
#include <memory>


#include <CL/sycl.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Isai preconditioner namespace.
 *
 * @ingroup isai
 */
namespace isai {


template <typename IndexType, typename Callback>
void forall_matching(const IndexType *fst, IndexType fst_size,
                     const IndexType *snd, IndexType snd_size,
                     Callback cb) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType, typename Callable>
void generic_generate(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *mtx,
                      matrix::Csr<ValueType, IndexType> *inverse_mtx,
                      IndexType *excess_rhs_ptrs, IndexType *excess_nz_ptrs,
                      Callable trs_solve) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void generate_tri_inverse(std::shared_ptr<const DefaultExecutor> exec,
                          const matrix::Csr<ValueType, IndexType> *mtx,
                          matrix::Csr<ValueType, IndexType> *inverse_mtx,
                          IndexType *excess_rhs_ptrs, IndexType *excess_nz_ptrs,
                          bool lower) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_TRI_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_excess_system(
    std::shared_ptr<const DefaultExecutor>,
    const matrix::Csr<ValueType, IndexType> *input,
    const matrix::Csr<ValueType, IndexType> *inverse,
    const IndexType *excess_rhs_ptrs, const IndexType *excess_nz_ptrs,
    matrix::Csr<ValueType, IndexType> *excess_system,
    matrix::Dense<ValueType> *excess_rhs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_EXCESS_SYSTEM_KERNEL);


template <typename ValueType, typename IndexType>
void scatter_excess_solution(
    std::shared_ptr<const DefaultExecutor>, const IndexType *excess_block_ptrs,
    const matrix::Dense<ValueType> *excess_solution,
    matrix::Csr<ValueType, IndexType> *inverse) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_SCATTER_EXCESS_SOLUTION_KERNEL);


}  // namespace isai
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
