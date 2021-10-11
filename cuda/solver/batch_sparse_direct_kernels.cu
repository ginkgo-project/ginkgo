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

#include "core/solver/batch_sparse_direct_kernels.hpp"


#include "cuda/base/cusolver_bindings.hpp"
#include "cuda/base/cusparse_bindings.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_sparse_direct {


template <typename ValueType>
void apply(std::shared_ptr<const CudaExecutor> exec,
           const matrix::BatchCsr<ValueType>* const a,
           const matrix::BatchDense<ValueType>* const b,
           matrix::BatchDense<ValueType>* const x,
           gko::log::BatchLogData<ValueType>& logdata)
{
    const size_type num_batches = a->get_num_batch_entries();
    // const auto nbatch = static_cast<int>(num_batches);
    const auto m = static_cast<int>(a->get_size().at()[0]);
    const auto nnz =
        static_cast<int>(a->get_num_stored_elements() / num_batches);
    const size_type b_stride = b->get_stride().at();
    const auto nrhs = static_cast<int>(b->get_size().at()[0]);

    auto handle = exec->get_cusolver_sp_handle();
    auto info = cusolver::create_csrqr_info();
    auto descr = cusparse::create_mat_descr();
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusolver::csrqr_batched_analysis(handle, m, m, nnz, descr,
                                     a->get_const_row_ptrs(),
                                     a->get_const_col_idxs(), info);
    size_type internal_bytes{};
    size_type workspace_bytes{};
    cusolver::csrqr_batched_buffer_info(
        handle, m, m, nnz, descr, a->get_const_values(),
        a->get_const_row_ptrs(), a->get_const_col_idxs(), num_batches, info,
        &internal_bytes, &workspace_bytes);

    const auto workspace = exec->alloc<unsigned char>(workspace_bytes);

    cusolver::csrqr_batched_solve(
        handle, m, m, nnz, descr, a->get_const_values(),
        a->get_const_row_ptrs(), a->get_const_col_idxs(), b->get_const_values(),
        x->get_values(), num_batches, info, workspace);

    exec->free(workspace);
    cusparse::destroy(descr);
    cusolver::destroy(info);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_SPARSE_DIRECT_APPLY_KERNEL);


}  // namespace batch_sparse_direct
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
