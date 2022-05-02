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


#include <algorithm>
#include <cassert>


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_ilu.hpp"
#include "reference/preconditioner/batch_trsv.hpp"


namespace gko {
namespace kernels {
namespace reference {
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
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto l_b = gko::batch::batch_entry(l_ub, batch);
        const auto u_b = gko::batch::batch_entry(u_ub, batch);
        generate(a_b.num_rows, a_b.row_ptrs, a_b.col_idxs, a_b.values,
                 l_b.row_ptrs, l_b.col_idxs, l_b.values, u_b.row_ptrs,
                 u_b.col_idxs, u_b.values);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_SPLIT_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_split(std::shared_ptr<const DefaultExecutor> exec,
                 const matrix::BatchCsr<ValueType, IndexType>* const l,
                 const matrix::BatchCsr<ValueType, IndexType>* const u,
                 const matrix::BatchDense<ValueType>* const r,
                 matrix::BatchDense<ValueType>* const z)
{
    auto lub = gko::kernels::host::get_batch_struct(l);
    auto uub = gko::kernels::host::get_batch_struct(u);
    auto rub = gko::kernels::host::get_batch_struct(r);
    auto zub = gko::kernels::host::get_batch_struct(z);
    using trsv_type = gko::kernels::host::batch_exact_trsv_split<ValueType>;
    using prec_type = gko::kernels::host::batch_ilu_split<ValueType, trsv_type>;
    prec_type prec(lub, uub, trsv_type());
    for (size_type batch = 0; batch < l->get_num_batch_entries(); ++batch) {
        const auto r_b = gko::batch::batch_entry(rub, batch);
        const auto z_b = gko::batch::batch_entry(zub, batch);
        prec.generate(batch, gko::batch_csr::BatchEntry<const ValueType>(),
                      nullptr);
        prec.apply(r_b, z_b);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_SPLIT_APPLY_KERNEL);


}  // namespace batch_ilu
}  // namespace reference
}  // namespace kernels
}  // namespace gko
