/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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
    IndexType* const sizes, IndexType* num_matches_per_row_for_each_csr_sys)
{
    const auto nrows = first_approx_inv->get_size()[0];
    const auto nnz_aiA = first_approx_inv->get_num_stored_elements();
    dim3 block(default_block_size);
    dim3 grid(ceildiv(nnz_aiA * default_subwarp_size, default_block_size));

    extract_dense_linear_sys_pattern_kernel<default_subwarp_size>
        <<<grid, block>>>(nrows, first_sys_csr->get_const_row_ptrs(),
                          first_sys_csr->get_const_col_idxs(),
                          first_approx_inv->get_const_row_ptrs(),
                          first_approx_inv->get_const_col_idxs(),
                          dense_mat_pattern, rhs_one_idxs, sizes,
                          num_matches_per_row_for_each_csr_sys);


    GKO_CUDA_LAST_IF_ERROR_THROW;
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


template <typename ValueType, typename IndexType>
void extract_csr_sys_pattern(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const int size,
    const matrix::Csr<ValueType, IndexType>* const first_approx_inv,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const csr_pattern)
{
    const auto nrows = first_approx_inv->get_size()[0];

    dim3 block(default_block_size);
    dim3 grid(ceildiv(size, default_block_size));

    extract_csr_sys_pattern_kernel<ValueType><<<grid, block>>>(
        lin_sys_row, first_approx_inv->get_const_row_ptrs(),
        first_approx_inv->get_const_col_idxs(),
        first_sys_csr->get_const_row_ptrs(),
        first_sys_csr->get_const_col_idxs(), csr_pattern->get_const_row_ptrs(),
        csr_pattern->get_col_idxs(), csr_pattern->get_values());

    GKO_CUDA_LAST_IF_ERROR_THROW;
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_EXTRACT_CSR_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void fill_batch_csr_sys_with_values(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<gko::remove_complex<ValueType>, IndexType>* const
        csr_pattern,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    matrix::BatchCsr<ValueType, IndexType>* const batch_csr_mats)
{
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto csr_nnz = csr_pattern->get_num_stored_elements();
    const auto sys_nnz = sys_csr->get_num_stored_elements() / nbatch;

    dim3 block(default_block_size);
    dim3 grid(ceildiv(nbatch * csr_nnz, default_block_size));

    fill_batch_csr_system_kernel<<<grid, block>>>(
        nbatch, csr_nnz, csr_pattern->get_const_values(), sys_nnz,
        as_cuda_type(sys_csr->get_const_values()),
        as_cuda_type(batch_csr_mats->get_values()));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_FILL_BATCH_CSR_SYSTEM_USING_PATTERN);


template <typename ValueType, typename IndexType>
void initialize_b_and_x_vectors(std::shared_ptr<const DefaultExecutor> exec,
                                const IndexType rhs_one_idx,
                                matrix::BatchDense<ValueType>* const b,
                                matrix::BatchDense<ValueType>* const x)
{
    const auto nbatch = b->get_num_batch_entries();
    const auto size = b->get_size().at(0)[0];

    dim3 block(default_block_size);
    dim3 grid(ceildiv(nbatch * size, default_block_size));

    initialize_b_and_x_vectors_kernel<<<grid, block>>>(
        nbatch, size, rhs_one_idx, as_cuda_type(b->get_values()),
        as_cuda_type(x->get_values()));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_INITIALIZE_B_AND_X);


template <typename ValueType, typename IndexType>
void write_large_sys_solution_to_inverse(
    std::shared_ptr<const DefaultExecutor> exec, const int lin_sys_row,
    const matrix::BatchDense<ValueType>* const x,
    matrix::BatchCsr<ValueType, IndexType>* const approx_inv)
{
    const auto nbatch = x->get_num_batch_entries();
    const auto size = x->get_size().at(0)[0];

    dim3 block(default_block_size);
    dim3 grid(ceildiv(nbatch * size, default_block_size));

    write_large_sys_solution_to_inverse_kernel<<<grid, block>>>(
        nbatch, lin_sys_row, size, as_cuda_type(x->get_const_values()),
        approx_inv->get_num_stored_elements() / nbatch,
        approx_inv->get_const_row_ptrs(),
        as_cuda_type(approx_inv->get_values()));

    GKO_CUDA_LAST_IF_ERROR_THROW;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_ISAI_WRITE_SOLUTION_TO_INVERSE);

}  // namespace batch_isai
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
