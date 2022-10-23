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

#include "core/preconditioner/batch_ilu_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/load_store.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_ilu_isai {
namespace {


constexpr size_type default_block_size = 256;

#include "common/cuda_hip/matrix/batch_vector_kernels.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_ilu_isai.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void apply_ilu_isai(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
                    const matrix::BatchCsr<ValueType, IndexType>* const l_inv,
                    const matrix::BatchCsr<ValueType, IndexType>* const u_inv,
                    const preconditioner::batch_ilu_isai_apply apply_type,
                    const matrix::BatchDense<ValueType>* const r,
                    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:batch_ilu_isai): change the code imported from
// preconditioner/batch_ilu if needed
//    const auto num_rows =
//        static_cast<int>(factored_matrix->get_size().at(0)[0]);
//    const auto nbatch = factored_matrix->get_num_batch_entries();
//    const auto factored_matrix_batch = get_batch_struct(factored_matrix);
//    using d_value_type = cuda_type<ValueType>;
//    using prec_type = batch_ilu_isai<d_value_type>;
//    prec_type prec(factored_matrix_batch, diag_locs);
//
//    batch_ilu_isai_apply<<<nbatch, default_block_size,
//                      prec_type::dynamic_work_size(
//                          num_rows,
//                          static_cast<int>(
//                              sys_matrix->get_num_stored_elements() / nbatch))
//                              *
//                          sizeof(ValueType)>>>(
//        prec, nbatch, num_rows, as_cuda_type(r->get_const_values()),
//        as_cuda_type(z->get_values()));
//}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ILU_ISAI_APPLY_KERNEL);

}  // namespace batch_ilu_isai
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
