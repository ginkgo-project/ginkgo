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

#include "core/preconditioner/batch_par_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
//#include "reference/preconditioner/batch_par_ilu.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace batch_par_ilu {


//#include "reference/preconditioner/batch_par_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void generate_common_pattern_to_fill_l_and_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_mat,
    const IndexType* const l_row_ptrs, const IndexType* const u_row_ptrs,
    IndexType* const l_col_holders,
    IndexType* const u_col_holders) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_GENERATE_COMMON_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_batch_l_and_batch_u(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const IndexType* const l_col_holders,
    const IndexType* const u_col_holders) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_INITIALIZE_BATCH_L_AND_BATCH_U);

template <typename ValueType, typename IndexType>
void compute_par_ilu0(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const int num_sweeps, const IndexType* const dependencies,
    const IndexType* const nz_ptrs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_COMPUTE_PARILU0_KERNEL);


template <typename ValueType, typename IndexType>
void apply_par_ilu0(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const l_factor,
    const matrix::BatchCsr<ValueType, IndexType>* const u_factor,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_PAR_ILU_APPLY_KERNEL);


}  // namespace batch_par_ilu
}  // namespace omp
}  // namespace kernels
}  // namespace gko
