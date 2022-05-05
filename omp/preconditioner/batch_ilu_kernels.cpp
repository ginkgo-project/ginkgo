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

#include "core/preconditioner/batch_ilu_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_ilu.hpp"


namespace gko {
namespace kernels {
namespace omp {
namespace batch_ilu {


#include "reference/preconditioner/batch_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void generate_split(std::shared_ptr<const DefaultExecutor> exec,
                    gko::preconditioner::batch_factorization_type,
                    const matrix::BatchCsr<ValueType, IndexType>* const a,
                    matrix::BatchCsr<ValueType, IndexType>* const l_factor,
                    matrix::BatchCsr<ValueType, IndexType>* const u_factor)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto l_ub = host::get_batch_struct(l_factor);
    const auto u_ub = host::get_batch_struct(u_factor);
#pragma omp parallel for firstprivate(a_ub, l_ub, u_ub)
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto l_b = gko::batch::batch_entry(l_ub, batch);
        const auto u_b = gko::batch::batch_entry(u_ub, batch);

        generate(a_b.num_rows, a_b.row_ptrs, a_b.col_idxs, a_b.values,
                 l_b.row_ptrs, l_b.col_idxs, l_b.values, u_b.row_ptrs,
                 u_b.col_idxs, u_b.values);
    }
    GKO_NOT_IMPLEMENTED;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_SPLIT_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_split(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::BatchCsr<ValueType, IndexType>* l,
                 const matrix::BatchCsr<ValueType, IndexType>* u,
                 const matrix::BatchDense<ValueType>* r,
                 matrix::BatchDense<ValueType>* z) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_SPLIT_APPLY_KERNEL);


}  // namespace batch_ilu
}  // namespace omp
}  // namespace kernels
}  // namespace gko
