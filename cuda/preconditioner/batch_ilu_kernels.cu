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
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_ilu {
namespace {


constexpr size_type default_block_size = 256;


#include "common/cuda_hip/preconditioner/batch_ilu.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_ilu_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_trsv.hpp.inc"


}  // namespace


template <typename ValueType, typename IndexType>
void generate_split(std::shared_ptr<const DefaultExecutor> exec,
                    gko::preconditioner::batch_factorization_type,
                    const matrix::BatchCsr<ValueType, IndexType>* const a,
                    matrix::BatchCsr<ValueType, IndexType>* const l,
                    matrix::BatchCsr<ValueType, IndexType>* const u)
{
    const auto num_rows = static_cast<int>(a->get_size().at(0)[0]);
    const auto nbatch = a->get_num_batch_entries();
    const auto nnz = static_cast<int>(a->get_num_stored_elements() / nbatch);
    const auto l_nnz = static_cast<int>(l->get_num_stored_elements() / nbatch);
    const auto u_nnz = static_cast<int>(u->get_num_stored_elements() / nbatch);
    generate<<<nbatch, default_block_size>>>(
        nbatch, num_rows, nnz, a->get_const_row_ptrs(), a->get_const_col_idxs(),
        as_cuda_type(a->get_const_values()), l_nnz, l->get_const_row_ptrs(),
        l->get_const_col_idxs(), as_cuda_type(l->get_values()), u_nnz,
        u->get_const_row_ptrs(), u->get_const_col_idxs(),
        as_cuda_type(u->get_values()));
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
    const auto num_rows = static_cast<int>(l->get_size().at(0)[0]);
    const auto nbatch = l->get_num_batch_entries();
    const auto l_ub = get_batch_struct(l);
    const auto u_ub = get_batch_struct(u);
    using d_value_type = cuda_type<ValueType>;
    using trsv_type = batch_exact_trsv_split<d_value_type>;
    using prec_type = batch_ilu_split<d_value_type, trsv_type>;
    prec_type prec(l_ub, u_ub, trsv_type());
    apply<<<nbatch, default_block_size>>>(prec, nbatch, num_rows,
                                          as_cuda_type(r->get_const_values()),
                                          as_cuda_type(z->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_SPLIT_APPLY_KERNEL);


}  // namespace batch_ilu
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
