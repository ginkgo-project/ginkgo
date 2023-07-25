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

#include "core/preconditioner/batch_isai_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "cuda/base/exception.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/load_store.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace batch_isai {
namespace {

constexpr size_type default_block_size = 256;
constexpr size_type default_subwarp_size = config::warp_size;
constexpr size_type max_grid_dim = 65535;

#include "common/cuda_hip/preconditioner/batch_isai.hpp.inc"
#include "common/cuda_hip/preconditioner/batch_isai_kernels.hpp.inc"

}  // namespace


template <typename ValueType, typename IndexType>
void extract_dense_linear_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    IndexType* const dense_mat_pattern, IndexType* const rhs_one_idxs,
    IndexType* const sizes)
{
    // std::cout << "Extract dense linear sys pattern - start" << std::endl;

    const auto nrows = first_approx_inv->get_size()[0];
    const auto nnz_aiA = first_approx_inv->get_num_stored_elements();
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nnz_aiA * default_subwarp_size, default_block_size));

    extract_dense_linear_sys_pattern_kernel<default_subwarp_size>
        <<<grid, block>>>(nrows, first_sys_csr->get_const_row_ptrs(),
                          first_sys_csr->get_const_col_idxs(),
                          first_approx_inv->get_const_row_ptrs(),
                          first_approx_inv->get_const_col_idxs(),
                          dense_mat_pattern, rhs_one_idxs, sizes);


    GKO_CUDA_LAST_IF_ERROR_THROW;

    // exec->synchronize();
    // std::cout << "Extract dense linear sys pattern - done" << std::endl;

    // using gko::preconditioner::batch_isai::row_size_limit;
    // gko::array<IndexType> dense_pattern_ref(exec->get_master(), nrows *
    // row_size_limit * row_size_limit);
    // exec->get_master()->copy_from(exec.get(), nrows * row_size_limit *
    // row_size_limit, dense_mat_pattern, dense_pattern_ref.get_data());

    // gko::array<IndexType> sizes_ref(exec->get_master(), nrows );
    // exec->get_master()->copy_from(exec.get(), nrows , sizes,
    // sizes_ref.get_data());

    // std::cout << "cuda dense pattern is: " << std::endl;
    // for(int row = 0; row < 3; row++)
    // {   std::cout << "corr to row: " << row << std::endl;
    //     for(int r = 0; r < sizes_ref.get_data()[row]; r++)
    //     {   std::cout << std::endl;
    //         for(int c = 0; c < sizes_ref.get_data()[row] ;c++)
    //         {
    //             std::cout << "dense(" << r << "," << c << ") : " <<
    //             dense_pattern_ref.get_const_data()[row* row_size_limit*
    //             row_size_limit  + r * row_size_limit + c] << "\n";
    //         }
    //     }
    // }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_DENSE_LINEAR_SYSTEM_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_values_dense_mat_and_solve(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const inv,
    const IndexType* const dense_mat_pattern,
    const IndexType* const rhs_one_idxs, const IndexType* const sizes,
    const gko::preconditioner::batch_isai_input_matrix_type&
        input_matrix_type_isai)
{
    const auto nbatch = inv->get_num_batch_entries();
    const auto nrows = static_cast<int>(inv->get_size().at(0)[0]);
    const auto A_nnz = sys_csr->get_num_stored_elements() / nbatch;
    const auto aiA_nnz = inv->get_num_stored_elements() / nbatch;

    dim3 block(default_block_size);
    auto grid_size =
        ceildiv(default_subwarp_size * nbatch * nrows, default_block_size);
    if (grid_size > max_grid_dim) {
        grid_size = max_grid_dim;
    }
    dim3 grid(grid_size);

    fill_values_dense_mat_and_solve_kernel<default_subwarp_size>
        <<<grid, block>>>(
            nbatch, nrows, A_nnz, as_cuda_type(sys_csr->get_const_values()),
            aiA_nnz, inv->get_const_row_ptrs(), as_cuda_type(inv->get_values()),
            dense_mat_pattern, rhs_one_idxs, sizes, input_matrix_type_isai);

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_VALUES_DENSE_MATRIX_AND_SOLVE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_isai(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
                const matrix::BatchCsr<ValueType, IndexType>* const approx_inv,
                const matrix::BatchDense<ValueType>* const r,
                matrix::BatchDense<ValueType>* const z)
{
    const auto num_rows = static_cast<int>(sys_mat->get_size().at(0)[0]);
    const auto nbatch = sys_mat->get_num_batch_entries();
    const auto approx_inv_batch = get_batch_struct(approx_inv);
    using d_value_type = cuda_type<ValueType>;
    using prec_type = batch_isai<d_value_type>;
    prec_type prec(approx_inv_batch);

    batch_isai_apply<<<nbatch, default_block_size,
                       prec_type::dynamic_work_size(
                           num_rows,
                           static_cast<int>(sys_mat->get_num_stored_elements() /
                                            nbatch)) *
                           sizeof(ValueType)>>>(
        prec, nbatch, num_rows, as_cuda_type(r->get_const_values()),
        as_cuda_type(z->get_values()));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_APPLY_KERNEL);

}  // namespace batch_isai
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
