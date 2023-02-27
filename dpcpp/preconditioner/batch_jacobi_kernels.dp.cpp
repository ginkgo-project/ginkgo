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
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "dpcpp/base/dim3.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_jacobi {

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z) GKO_NOT_IMPLEMENTED;


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL);


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const blocks_cumulative_storage)
{
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(num_blocks);

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1) {
                             auto gid = item_ct1.get_global_linear_id();
                             const auto bsize =
                                 block_pointers[gid + 1] - block_pointers[gid];
                             blocks_cumulative_storage[gid] = bsize * bsize;
                         });
    });
    components::prefix_sum(exec, blocks_cumulative_storage, num_blocks + 1);
}

template void compute_cumulative_block_storage<int>(
    std::shared_ptr<const DefaultExecutor>, const size_type, const int32* const,
    int32* const);


template <typename IndexType>
void find_row_is_part_of_which_block(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const row_part_of_which_block_info) GKO_NOT_IMPLEMENTED;

// instantiate for index type int32
template void find_row_is_part_of_which_block<int>(
    std::shared_ptr<const DefaultExecutor>, const size_type, const int32* const,
    int32* const);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const row_part_of_which_block_info,
    IndexType* const blocks_pattern) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const uint32 max_block_size, const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern,
    ValueType* const blocks) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);

template <typename ValueType, typename IndexType>
void transpose_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec, const size_type nbatch,
    const size_type nrows, const size_type num_blocks,
    const uint32 max_block_size, const IndexType* const block_pointers,
    const ValueType* const blocks_array,
    const gko::preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const row_part_of_which_block_info,
    ValueType* const out_blocks_array,
    const bool to_conjugate) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_TRANSPOSE_KERNEL);

}  // namespace batch_jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
