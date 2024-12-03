// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/batch_jacobi_kernels.hpp"

#include <ginkgo/core/base/exception_helpers.hpp>

#include "core/base/batch_struct.hpp"
#include "core/base/utils.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/batch_struct.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "dpcpp/base/batch_struct.hpp"
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/dpct.hpp"
#include "dpcpp/matrix/batch_struct.hpp"
#include "dpcpp/preconditioner/batch_jacobi_kernels.hpp"
#include "dpcpp/preconditioner/jacobi_common.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
namespace batch_jacobi {
namespace {


using batch_jacobi_dpcpp_compiled_max_block_sizes =
    gko::kernels::dpcpp::jacobi::compiled_kernels;


}  // namespace


template <typename IndexType>
void compute_cumulative_block_storage(
    std::shared_ptr<const DefaultExecutor> exec, const size_type num_blocks,
    const IndexType* block_pointers, IndexType* blocks_cumulative_offsets)
{
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_blocks, [=](auto id) {
            const auto bsize = block_pointers[id + 1] - block_pointers[id];
            blocks_cumulative_offsets[id] = bsize * bsize;
        });
    });
    components::prefix_sum_nonnegative(exec, blocks_cumulative_offsets,
                                       num_blocks + 1);
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_CUMULATIVE_BLOCK_STORAGE);


template <typename IndexType>
void find_row_block_map(std::shared_ptr<const DefaultExecutor> exec,
                        const size_type num_blocks,
                        const IndexType* block_pointers,
                        IndexType* map_block_to_row)
{
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(num_blocks, [=](auto id) {
            for (int i = block_pointers[id]; i < block_pointers[id + 1]; i++)
                map_block_to_row[i] = id;
        });
    });
}

GKO_INSTANTIATE_FOR_INT32_TYPE(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_FIND_ROW_BLOCK_MAP);


template <typename ValueType, typename IndexType>
void extract_common_blocks_pattern(
    std::shared_ptr<const DefaultExecutor> exec,
    const gko::matrix::Csr<ValueType, IndexType>* first_sys_csr,
    const size_type num_blocks, const IndexType* cumulative_block_storage,
    const IndexType* block_pointers, const IndexType* map_block_row,
    IndexType* blocks_pattern)
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

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[intel::reqd_sub_group_size(subgroup_size)]] {
                    batch_single_kernels::extract_common_block_pattern_kernel(
                        static_cast<int>(nrows), row_ptrs, col_idxs, num_blocks,
                        cumulative_block_storage, block_pointers, map_block_row,
                        blocks_pattern, item_ct1);
                });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE_WITH_HALF(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL);


namespace {


template <int compiled_max_block_size, typename ValueType, typename IndexType>
void compute_block_jacobi_helper(
    syn::value_list<int, compiled_max_block_size>,
    const batch::matrix::Csr<ValueType, IndexType>* const sys_csr,
    const size_type num_blocks, const IndexType* const cumulative_block_storage,
    const IndexType* const block_pointers,
    const IndexType* const blocks_pattern, ValueType* const blocks,
    std::shared_ptr<const DpcppExecutor> exec)
{
    //    constexpr int subwarp_size =
    //        gko::kernels::dpcpp::jacobi::get_larger_power(compiled_max_block_size);
    // TODO: Find a way to allow smaller block_sizes (<16)

    constexpr int subgroup_size = config::warp_size;
    auto device = exec->get_queue()->get_device();
    auto group_size =
        device.get_info<sycl::info::device::max_work_group_size>();

    const auto nbatch = sys_csr->get_num_batch_items();
    const auto nrows = sys_csr->get_common_size()[0];
    const auto nnz = sys_csr->get_num_stored_elements() / nbatch;
    const auto sys_csr_values = sys_csr->get_const_values();

    dim3 block(group_size);
    dim3 grid(ceildiv(num_blocks * nbatch * subgroup_size, group_size));

    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[intel::reqd_sub_group_size(subgroup_size)]] {
                    batch_single_kernels::compute_block_jacobi_kernel(
                        nbatch, static_cast<int>(nnz), sys_csr_values,
                        num_blocks, cumulative_block_storage, block_pointers,
                        blocks_pattern, blocks, item_ct1);
                });
    });
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_block_jacobi_helper,
                                    compute_block_jacobi_helper);

}  // anonymous namespace


template <typename ValueType, typename IndexType>
void compute_block_jacobi(
    std::shared_ptr<const DefaultExecutor> exec,
    const batch::matrix::Csr<ValueType, IndexType>* sys_csr,
    const uint32 user_given_max_block_size, const size_type num_blocks,
    const IndexType* cumulative_block_storage, const IndexType* block_pointers,
    const IndexType* blocks_pattern, ValueType* blocks)
{
    select_compute_block_jacobi_helper(
        batch_jacobi_dpcpp_compiled_max_block_sizes(),
        [&](int compiled_block_size) {
            return user_given_max_block_size <= compiled_block_size;
        },
        syn::value_list<int>(), syn::type_list<>(), sys_csr, num_blocks,
        cumulative_block_storage, block_pointers, blocks_pattern, blocks, exec);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INT32_TYPE_WITH_HALF(
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL);


}  // namespace batch_jacobi
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
