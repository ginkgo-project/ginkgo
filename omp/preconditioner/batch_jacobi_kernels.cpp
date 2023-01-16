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

#include "core/preconditioner/batch_jacobi_kernels.hpp"


#include <ginkgo/core/matrix/batch_csr.hpp>


#include "core/matrix/batch_struct.hpp"
#include "reference/matrix/batch_struct.hpp"
#include "reference/preconditioner/batch_block_jacobi.hpp"
#include "reference/preconditioner/batch_scalar_jacobi.hpp"

namespace gko {
namespace kernels {
namespace omp {

namespace batch_jacobi {

namespace {


template <typename BatchMatrixType, typename PrecType, typename ValueType>
void apply_jacobi(const BatchMatrixType& sys_mat_batch, PrecType& prec,
                  const gko::batch_dense::UniformBatch<const ValueType>& rub,
                  const gko::batch_dense::UniformBatch<ValueType>& zub)
{
#pragma omp parallel for
    for (size_type batch_id = 0; batch_id < sys_mat_batch.num_batch;
         batch_id++) {
        const auto sys_mat_entry =
            gko::batch::batch_entry(sys_mat_batch, batch_id);
        const auto r_b = gko::batch::batch_entry(rub, batch_id);
        const auto z_b = gko::batch::batch_entry(zub, batch_id);

        const auto work_arr_size = PrecType::dynamic_work_size(
            sys_mat_batch.num_rows, sys_mat_batch.num_nnz);
        std::vector<ValueType> work(work_arr_size);

        prec.generate(batch_id, sys_mat_entry, work.data());
        prec.apply(r_b, z_b);
    }
}
// Note: Do not change the ordering

#include "reference/preconditioner/batch_jacobi_kernels.hpp.inc"

}  // unnamed namespace


template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const gko::OmpExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const ValueType* blocks_array, const IndexType* block_ptrs,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto sys_mat_batch = gko::kernels::host::get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(sys_mat_batch, num_blocks, max_block_size,
                              blocks_array, block_ptrs, r, z);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const gko::OmpExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const ValueType* blocks_array, const IndexType* block_ptrs,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto sys_mat_batch = gko::kernels::host::get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(sys_mat_batch, num_blocks, max_block_size,
                              blocks_array, block_ptrs, r, z);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL);


template <typename ValueType>
void batch_jacobi_apply(std::shared_ptr<const gko::OmpExecutor> exec,
                        const matrix::BatchEll<ValueType>* const a,
                        const matrix::BatchDense<ValueType>* const b,
                        matrix::BatchDense<ValueType>* const x)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const int local_size_bytes =
        host::BatchScalarJacobi<ValueType>::dynamic_work_size(a_ub.num_rows,
                                                              a_ub.num_nnz) *
        sizeof(ValueType);
    using byte = unsigned char;

#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        array<byte> local_space(exec, local_size_bytes);
        host::BatchScalarJacobi<ValueType> prec;

        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);

        const auto prec_work =
            reinterpret_cast<ValueType*>(local_space.get_data());
        prec.generate(batch, a_b, prec_work);
        prec.apply(b_b, x_b);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_SCALAR_JACOBI_ELL_APPLY_KERNEL);


template <typename ValueType>
void batch_jacobi_apply(std::shared_ptr<const gko::OmpExecutor> exec,
                        const matrix::BatchCsr<ValueType>* const a,
                        const matrix::BatchDense<ValueType>* const b,
                        matrix::BatchDense<ValueType>* const x)
{
    const auto a_ub = host::get_batch_struct(a);
    const auto b_ub = host::get_batch_struct(b);
    const auto x_ub = host::get_batch_struct(x);
    const int local_size_bytes =
        host::BatchScalarJacobi<ValueType>::dynamic_work_size(a_ub.num_rows,
                                                              a_ub.num_nnz) *
        sizeof(ValueType);
    using byte = unsigned char;

#pragma omp parallel for
    for (size_type batch = 0; batch < a->get_num_batch_entries(); ++batch) {
        array<byte> local_space(exec, local_size_bytes);
        host::BatchScalarJacobi<ValueType> prec;

        const auto a_b = gko::batch::batch_entry(a_ub, batch);
        const auto b_b = gko::batch::batch_entry(b_ub, batch);
        const auto x_b = gko::batch::batch_entry(x_ub, batch);

        const auto prec_work =
            reinterpret_cast<ValueType*>(local_space.get_data());
        prec.generate(batch, a_b, prec_work);
        prec.apply(b_b, x_b);
    }
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_BATCH_SCALAR_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const uint32 max_block_size, const size_type num_blocks,
    const IndexType* const block_pointers, IndexType* const blocks_pattern)
{
#pragma omp parallel for
    for (size_type k = 0; k < num_blocks; k++) {
        extract_block_pattern_impl(k, first_sys_csr, max_block_size,
                                   block_pointers, blocks_pattern);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const size_type num_blocks, const uint32 max_block_size,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto A_batch = host::get_batch_struct(sys_csr);

#pragma omp parallel for
    for (size_type i = 0; i < nbatch * num_blocks; i++) {
        size_type block_idx = i % num_blocks;
        size_type batch_idx = i / num_blocks;

        const auto A_entry = gko::batch::batch_entry(A_batch, batch_idx);
        compute_block_jacobi_impl(batch_idx, block_idx, A_entry, num_blocks,
                                  max_block_size, block_pointers,
                                  blocks_pattern, blocks);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);

}  // namespace batch_jacobi
}  // namespace omp
}  // namespace kernels
}  // namespace gko
