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


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/matrix/batch_struct.hpp"
#include "dpcpp/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_jacobi {

namespace {

#include "dpcpp/preconditioner/batch_block_jacobi.hpp.inc"
#include "dpcpp/preconditioner/batch_jacobi.hpp.inc"
#include "dpcpp/preconditioner/batch_scalar_jacobi.hpp.inc"

using batch_jacobi_dpcpp_compiled_max_block_sizes =
    gko::kernels::dpcpp::jacobi::compiled_kernels;

}  // namespace

namespace {

template <typename BatchMatrixType, typename IndexType, typename ValueType>
void batch_jacobi_apply_helper(
    std::shared_ptr<const DpcppExecutor> exec,
    const BatchMatrixType& sys_mat_batch, const size_type num_blocks,
    const uint32 max_block_size,
    const gko::preconditioner::batched_jacobi_blocks_storage_scheme<int>&
        storage_scheme,
    const int* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto nbatch = sys_mat_batch.num_batch;
    const auto nrows = sys_mat_batch.num_rows;

    const auto r_ub = get_batch_struct(r);
    const auto z_ub = get_batch_struct(z);

    constexpr int subgroup_size = config::warp_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();
    size_type slm_size = device.get_info<sycl::info::device::local_mem_size>();

    const dim3 block(group_size);
    const dim3 grid(nbatch);

    const auto r_values = r->get_const_values();
    auto z_values = z->get_values();
    if (max_block_size == 1u) {
        const auto shared_size =
            BatchScalarJacobi<ValueType>::dynamic_work_size(
                sys_mat_batch.num_rows, sys_mat_batch.num_nnz);
        GKO_ASSERT(shared_size * sizeof(ValueType) <= slm_size);
        auto prec_scalar_jacobi = BatchScalarJacobi<ValueType>();

        (exec->get_queue())->submit([&](sycl::handler& cgh) {
            sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_storage(sycl::range<1>(shared_size), cgh);
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
                    auto batch_id = item_ct1.get_group_linear_id();
                    batch_scalar_jacobi_apply(
                        prec_scalar_jacobi, sys_mat_batch, batch_id, nrows,
                        r_values, z_values,
                        static_cast<ValueType*>(slm_storage.get_pointer()),
                        item_ct1);
                });
        });
    } else {
        const auto shared_size = BatchBlockJacobi<ValueType>::dynamic_work_size(
            sys_mat_batch.num_rows, sys_mat_batch.num_nnz);
        GKO_ASSERT(shared_size * sizeof(ValueType) <= slm_size);
        auto prec_block_jacobi = BatchBlockJacobi<ValueType>(
            max_block_size, num_blocks, storage_scheme,
            cumulative_block_storage, blocks_array, block_ptrs,
            row_part_of_which_block_info);

        (exec->get_queue())->submit([&](sycl::handler& cgh) {
            sycl::accessor<ValueType, 1, sycl::access_mode::read_write,
                           sycl::access::target::local>
                slm_storage(sycl::range<1>(shared_size), cgh);
            cgh.parallel_for(
                sycl_nd_range(grid, block), [=
            ](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(
                                                subgroup_size)]] {
                    auto batch_id = item_ct1.get_group_linear_id();
                    batch_block_jacobi_apply(
                        prec_block_jacobi, batch_id, nrows, r_values, z_values,
                        static_cast<ValueType*>(slm_storage.get_pointer()),
                        item_ct1);
                });
        });
    }
}

}  // namespace

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const gko::preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto a_ub = get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(exec, a_ub, num_blocks, max_block_size,
                              storage_scheme, cumulative_block_storage,
                              blocks_array, block_ptrs,
                              row_part_of_which_block_info, r, z);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL);

template <typename ValueType, typename IndexType>
void batch_jacobi_apply(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchEll<ValueType, IndexType>* const sys_mat,
    const size_type num_blocks, const uint32 max_block_size,
    const gko::preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const ValueType* const blocks_array, const IndexType* const block_ptrs,
    const IndexType* const row_part_of_which_block_info,
    const matrix::BatchDense<ValueType>* const r,
    matrix::BatchDense<ValueType>* const z)
{
    const auto a_ub = get_batch_struct(sys_mat);
    batch_jacobi_apply_helper(exec, a_ub, num_blocks, max_block_size,
                              storage_scheme, cumulative_block_storage,
                              blocks_array, block_ptrs,
                              row_part_of_which_block_info, r, z);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL);


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DpcppExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const blocks_cumulative_storage)
{
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_blocks, [=](auto id) {
            const auto bsize = block_pointers[id + 1] - block_pointers[id];
            blocks_cumulative_storage[id] = bsize * bsize;
        });
    });
    exec->get_queue()->wait();
    components::prefix_sum_nonnegative(exec, blocks_cumulative_storage,
                                       num_blocks + 1);
}

template void compute_cumulative_block_storage<int>(
    std::shared_ptr<const DpcppExecutor>, const size_type, const int32* const,
    int32* const);


template <typename IndexType>
void find_row_is_part_of_which_block(
    std::shared_ptr<const DpcppExecutor> exec, const size_type num_blocks,
    const IndexType* const block_pointers,
    IndexType* const row_part_of_which_block_info)
{
    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_blocks, [=](auto id) {
            for (int i = block_pointers[id]; i < block_pointers[id + 1]; i++)
                row_part_of_which_block_info[i] = id;
        });
    });
}

// instantiate for index type int32
template void find_row_is_part_of_which_block<int>(
    std::shared_ptr<const DpcppExecutor>, const size_type, const int32* const,
    int32* const);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* const first_sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const row_part_of_which_block_info,
    IndexType* const blocks_pattern)
{
    const auto nrows = first_sys_csr->get_size()[0];
    constexpr int subgroup_size = config::warp_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const dim3 block(group_size);
    const dim3 grid(ceildiv(nrows * subgroup_size, group_size));

    const auto row_ptrs = first_sys_csr->get_const_row_ptrs();
    const auto col_idxs = first_sys_csr->get_const_col_idxs();

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                            subgroup_size)]] {
                extract_common_block_pattern_kernel(
                    static_cast<int>(nrows), row_ptrs, col_idxs, num_blocks,
                    storage_scheme, cumulative_block_storage, block_pointers,
                    row_part_of_which_block_info, blocks_pattern, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


namespace {

template <int compiled_max_block_size, typename ValueType, typename IndexType>
void compute_block_jacobi_helper(
    syn::value_list<int, compiled_max_block_size>,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks,
    std::shared_ptr<const DpcppExecutor> exec)
{
    //    constexpr int subwarp_size =
    //        gko::kernels::dpcpp::jacobi::get_larger_power(compiled_max_block_size);
    // TODO: Find the way to allow smaller block_sizes (<16)

    constexpr int subgroup_size = config::warp_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const auto nbatch = sys_csr->get_num_batch_entries();
    const auto nrows = sys_csr->get_size().at(0)[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;
    const auto sys_csr_values = sys_csr->get_const_values();

    dim3 block(group_size);
    dim3 grid(ceildiv(num_blocks * nbatch * subgroup_size, group_size));

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                            subgroup_size)]] {
                compute_block_jacobi_kernel(
                    nbatch, static_cast<int>(nnz), sys_csr_values, num_blocks,
                    storage_scheme, cumulative_block_storage, block_pointers,
                    blocks_pattern, blocks, item_ct1);
            });
    });
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_block_jacobi_helper,
                                    compute_block_jacobi_helper);

}  // anonymous namespace

template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::BatchCsr<ValueType, IndexType>* const sys_csr,
    const uint32 user_given_max_block_size, const size_type num_blocks,
    const preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks)
{
    select_compute_block_jacobi_helper(
        batch_jacobi_dpcpp_compiled_max_block_sizes(),
        [&](int compiled_block_size) {
            return user_given_max_block_size <= compiled_block_size;
        },
        syn::value_list<int>(), syn::type_list<>(), sys_csr, num_blocks,
        storage_scheme, cumulative_block_storage, block_pointers,
        blocks_pattern, blocks, exec);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);

template <typename ValueType, typename IndexType>
void transpose_block_jacobi(
    std::shared_ptr<const DpcppExecutor> exec, const size_type nbatch,
    const size_type nrows, const size_type num_blocks,
    const uint32 max_block_size, const IndexType* const block_pointers,
    const ValueType* const blocks_array,
    const gko::preconditioner::batched_jacobi_blocks_storage_scheme<IndexType>&
        storage_scheme,
    const IndexType* const cumulative_block_storage,
    const IndexType* const row_part_of_which_block_info,
    ValueType* const out_blocks_array, const bool to_conjugate)
{
    constexpr int subgroup_size = config::warp_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    dim3 block(group_size);
    dim3 grid(ceildiv(nrows * nbatch * subgroup_size, group_size));

    (exec->get_queue())->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block), [=
        ](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(
                                            subgroup_size)]] {
                transpose_block_jacobi_kernel(
                    nbatch, static_cast<int>(nrows), num_blocks, block_pointers,
                    blocks_array, storage_scheme, cumulative_block_storage,
                    row_part_of_which_block_info, out_blocks_array,
                    to_conjugate, item_ct1);
            });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE_AND_INT32_INDEX(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_TRANSPOSE_KERNEL);

}  // namespace batch_jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
