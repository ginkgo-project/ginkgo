// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/dpl/algorithm>

#include "core/factorization/lu_kernels.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>

#include <sycl/ext/oneapi/experimental/builtins.hpp>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/base/allocator.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "dpcpp/base/helper.hpp"
#include "dpcpp/base/onedpl.hpp"
#include "dpcpp/base/types.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/syncfree.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


#ifdef __SYCL_DEVICE_ONLY__
#define __SYCL_CONSTANT_AS __attribute__((opencl_constant))
#else
#define __SYCL_CONSTANT_AS
#endif


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The LU namespace.
 *
 * @ingroup factor
 */
namespace lu_factorization {


constexpr static int default_block_size = 512;


namespace kernel {


template <typename ValueType, typename IndexType>
void initialize(const IndexType* __restrict__ mtx_row_ptrs,
                const IndexType* __restrict__ mtx_cols,
                const ValueType* __restrict__ mtx_vals,
                const IndexType* __restrict__ factor_row_ptrs,
                const IndexType* __restrict__ factor_cols,
                const IndexType* __restrict__ factor_storage_offsets,
                const int32* __restrict__ factor_storage,
                const int64* __restrict__ factor_row_descs,
                ValueType* __restrict__ factor_vals,
                IndexType* __restrict__ diag_idxs, size_type num_rows,
                sycl::nd_item<3> item_ct1)
{
    const auto row = thread::get_subwarp_id_flat<config::warp_size>(item_ct1);
    if (row >= num_rows) {
        return;
    }
    const auto warp = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    // first zero out this row of the factor
    const auto factor_begin = factor_row_ptrs[row];
    const auto factor_end = factor_row_ptrs[row + 1];
    const auto lane = static_cast<int>(warp.thread_rank());
    for (auto nz = factor_begin + lane; nz < factor_end;
         nz += config::warp_size) {
        factor_vals[nz] = zero<ValueType>();
    }
    warp.sync();
    // then fill in the values from mtx
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        factor_row_ptrs, factor_cols,      factor_storage_offsets,
        factor_storage,  factor_row_descs, row};
    const auto row_begin = mtx_row_ptrs[row];
    const auto row_end = mtx_row_ptrs[row + 1];
    for (auto nz = row_begin + lane; nz < row_end; nz += config::warp_size) {
        const auto col = mtx_cols[nz];
        const auto val = mtx_vals[nz];
        factor_vals[lookup.lookup_unsafe(col) + factor_begin] = val;
    }
    if (lane == 0) {
        diag_idxs[row] = lookup.lookup_unsafe(row) + factor_begin;
    }
}

GKO_ENABLE_DEFAULT_HOST(initialize, initialize);


template <bool full_fillin, typename ValueType, typename IndexType>
void factorize(
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    const IndexType* __restrict__ diag_idxs, ValueType* __restrict__ vals,
    syncfree_storage dep_storage, size_type num_rows, sycl::nd_item<3> item_ct1,
    typename syncfree_scheduler<default_block_size, config::warp_size,
                                IndexType>::shared_storage& sh_dep_storage)
{
    // static const __SYCL_CONSTANT_AS char FMT[] = "%d n: %d\n";
    // sycl::ext::oneapi::experimental::printf(FMT, 0,
    // item_ct1.get_local_id(2));
    using scheduler_t =
        syncfree_scheduler<default_block_size, config::warp_size, IndexType>;
    scheduler_t scheduler(dep_storage, sh_dep_storage, item_ct1);
    const auto row = scheduler.get_work_id();
    auto sg = item_ct1.get_sub_group();
    if (row >= num_rows) {
        return;
    }
    const auto warp = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    const auto lane = warp.thread_rank();
    const auto row_begin = row_ptrs[row];
    const auto row_diag = diag_idxs[row];
    const auto row_end = row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        row_ptrs, cols,      storage_offsets,
        storage,  row_descs, static_cast<size_type>(row)};
    // if (item_ct1.get_local_id(2) % config::warp_size == 0) {
    //     sycl::ext::oneapi::experimental::printf(FMT, 2, row);
    // }
    // for each lower triangular entry: eliminate with corresponding row
    for (auto lower_nz = row_begin; lower_nz < row_diag; lower_nz++) {
        const auto dep = cols[lower_nz];
        // we can load the value before synchronizing because the following
        // updates only go past the diagonal of the dependency row, i.e. at
        // least column dep + 1
        const auto diag_idx = diag_idxs[dep];
        const auto dep_end = row_ptrs[dep + 1];
        scheduler.wait(dep);
        sg.barrier();
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        const auto val = vals[lower_nz];
        const auto diag = vals[diag_idx];
        const auto scale = val / diag;
        if (lane == 0) {
            vals[lower_nz] = scale;
        }
        // subtract all entries past the diagonal
        for (auto upper_nz = diag_idx + 1 + lane; upper_nz < dep_end;
             upper_nz += config::warp_size) {
            const auto upper_col = cols[upper_nz];
            const auto upper_val = vals[upper_nz];
            if constexpr (full_fillin) {
                const auto output_pos =
                    lookup.lookup_unsafe(upper_col) + row_begin;
                vals[output_pos] -= scale * upper_val;
            } else {
                const auto pos = lookup[upper_col];
                if (pos != invalid_index<IndexType>()) {
                    vals[row_begin + pos] -= scale * upper_val;
                }
            }
        }
    }
    scheduler.mark_ready();
    sg.barrier();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
}

template <bool full_fillin, typename ValueType, typename IndexType>
void factorize(dim3 grid, dim3 block, gko::size_type, sycl::queue* queue,
               const IndexType* row_ptrs, const IndexType* cols,
               const IndexType* storage_offsets, const int32* storage,
               const int64* row_descs, const IndexType* diag_idxs,
               ValueType* vals, syncfree_storage dep_storage,
               size_type num_rows)
{
    /*
using size_query = sycl::info::kernel::max_work_group_size;
using num_query = sycl::info::kernel::max_num_work_groups;

auto bundle = sycl::get_kernel_bundle(queue->get_context());
auto kernel = bundle.get_kernel<class kernel::factorize<full_filllin, ValueType,
IndexType>>(); auto wg_size = kernel.get_info<size_query>(*queue); auto num_wg =
kernel.get_info<num_query>(*queue, wg_size); gtd::cout << "wg_size " << wg_size
<< " num_wg " << num_wg << std::endl;*/
    queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<
            typename syncfree_scheduler<default_block_size, config::warp_size,
                                        IndexType>::shared_storage,
            0>
            sh_dep_storage_acc_ct1(cgh);
        cgh.parallel_for(sycl_nd_range(grid, block),
                         [=](sycl::nd_item<3> item_ct1)
                             [[sycl::reqd_sub_group_size(config::warp_size)]] {
                                 factorize<full_fillin>(
                                     row_ptrs, cols, storage_offsets, storage,
                                     row_descs, diag_idxs, vals, dep_storage,
                                     num_rows, item_ct1,
                                     *sh_dep_storage_acc_ct1.get_pointer());
                             });
    });
};


template <typename ValueType, typename IndexType>
void symbolic_factorize_simple(
    const IndexType* __restrict__ mtx_row_ptrs,
    const IndexType* __restrict__ mtx_cols,
    const IndexType* __restrict__ factor_row_ptrs,
    const IndexType* __restrict__ factor_cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    IndexType* __restrict__ diag_idxs, ValueType* __restrict__ factor_vals,
    IndexType* __restrict__ out_row_nnz, syncfree_storage dep_storage,
    size_type num_rows, sycl::nd_item<3> item_ct1,
    typename syncfree_scheduler<default_block_size, config::warp_size,
                                IndexType>::shared_storage& sh_dep_storage)
{
    using scheduler_t =
        syncfree_scheduler<default_block_size, config::warp_size, IndexType>;
    scheduler_t scheduler(dep_storage, sh_dep_storage, item_ct1);
    const auto row = scheduler.get_work_id();
    if (row >= num_rows) {
        return;
    }
    const auto warp = group::tiled_partition<config::warp_size>(
        group::this_thread_block(item_ct1));
    const auto lane = warp.thread_rank();
    const auto factor_begin = factor_row_ptrs[row];
    const auto factor_end = factor_row_ptrs[row + 1];
    const auto mtx_begin = mtx_row_ptrs[row];
    const auto mtx_end = mtx_row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        factor_row_ptrs, factor_cols, storage_offsets,
        storage,         row_descs,   static_cast<size_type>(row)};
    const auto row_diag = lookup.lookup_unsafe(row) + factor_begin;
    // fill with zeros first
    for (auto nz = factor_begin + lane; nz < factor_end;
         nz += config::warp_size) {
        factor_vals[nz] = zero<float>();
    }
    warp.sync();
    // then fill in the system matrix
    for (auto nz = mtx_begin + lane; nz < mtx_end; nz += config::warp_size) {
        const auto col = mtx_cols[nz];
        factor_vals[lookup.lookup_unsafe(col) + factor_begin] = one<float>();
    }
    // finally set diagonal and store diagonal index
    if (lane == 0) {
        diag_idxs[row] = row_diag;
        factor_vals[row_diag] = one<float>();
    }
    warp.sync();
    // for each lower triangular entry: eliminate with corresponding row
    for (auto lower_nz = factor_begin; lower_nz < row_diag; lower_nz++) {
        const auto dep = factor_cols[lower_nz];
        const auto dep_end = factor_row_ptrs[dep + 1];
        scheduler.wait(dep);
        sycl::atomic_fence(sycl::memory_order::acq_rel,
                           sycl::memory_scope::device);
        // read the diag entry after we are sure it was written.
        const auto diag_idx = diag_idxs[dep];
        if (factor_vals[lower_nz] == one<float>()) {
            // eliminate with upper triangle/entries past the diagonal
            for (auto upper_nz = diag_idx + 1 + lane; upper_nz < dep_end;
                 upper_nz += config::warp_size) {
                const auto upper_col = factor_cols[upper_nz];
                const auto upper_val = factor_vals[upper_nz];
                const auto output_pos =
                    lookup.lookup_unsafe(upper_col) + factor_begin;
                if (upper_val == one<float>()) {
                    factor_vals[output_pos] = one<float>();
                }
            }
        }
    }
    scheduler.mark_ready();
    sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
    IndexType row_nnz{};
    for (auto nz = factor_begin + lane; nz < factor_end;
         nz += config::warp_size) {
        row_nnz += factor_vals[nz] == one<float>() ? 1 : 0;
    }
    row_nnz = ::gko::kernels::dpcpp::reduce(
        warp, row_nnz, [](auto a, auto b) { return a + b; });
    if (lane == 0) {
        out_row_nnz[row] = row_nnz;
    }
}

template <typename ValueType, typename IndexType>
void symbolic_factorize_simple(
    dim3 grid, dim3 block, gko::size_type, sycl::queue* queue,
    const IndexType* mtx_row_ptrs, const IndexType* mtx_cols,
    const IndexType* factor_row_ptrs, const IndexType* factor_cols,
    const IndexType* storage_offsets, const int32* storage,
    const int64* row_descs, IndexType* diag_idxs, ValueType* factor_vals,
    IndexType* out_row_nnz, syncfree_storage dep_storage, size_type num_rows)
{
    queue->submit([&](sycl::handler& cgh) {
        sycl::local_accessor<
            typename syncfree_scheduler<default_block_size, config::warp_size,
                                        IndexType>::shared_storage,
            0>
            sh_dep_storage_acc_ct1(cgh);
        cgh.parallel_for(
            sycl_nd_range(grid, block),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(config::warp_size)]] {
                    symbolic_factorize_simple(
                        mtx_row_ptrs, mtx_cols, factor_row_ptrs, factor_cols,
                        storage_offsets, storage, row_descs, diag_idxs,
                        factor_vals, out_row_nnz, dep_storage, num_rows,
                        item_ct1, *sh_dep_storage_acc_ct1.get_pointer());
                });
    });
};


}  // namespace kernel


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    const auto num_rows = mtx->get_size()[0];
    if (num_rows > 0) {
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::initialize(
            num_blocks, default_block_size, 0, exec->get_queue(),
            mtx->get_const_row_ptrs(), mtx->get_const_col_idxs(),
            as_device_type(mtx->get_const_values()),
            factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
            factor_lookup_offsets, factor_lookup_storage, factor_lookup_descs,
            as_device_type(factors->get_values()), diag_idxs, num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               matrix::Csr<ValueType, IndexType>* factors, bool full_fillin,
               array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        std::cout << "num_blocks " << num_blocks << std::endl;
        std::cout << "multiprocessor " << exec->get_num_computing_units()
                  << " num_warp " << exec->get_num_subgroups() << std::endl;
        if (full_fillin) {
            kernel::factorize<true>(
                num_blocks, default_block_size, 0, exec->get_queue(),
                factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
                lookup_offsets, lookup_storage, lookup_descs, diag_idxs,
                as_device_type(factors->get_values()), storage, num_rows);
        } else {
            kernel::factorize<false>(
                num_blocks, default_block_size, 0, exec->get_queue(),
                factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
                lookup_offsets, lookup_storage, lookup_descs, diag_idxs,
                as_device_type(factors->get_values()), storage, num_rows);
        }
    }
}
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_LU_FACTORIZE);


template <typename IndexType>
void symbolic_factorize_simple(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs, const IndexType* lookup_offsets,
    const int64* lookup_descs, const int32* lookup_storage,
    matrix::Csr<float, IndexType>* factors, IndexType* out_row_nnz)
{
    const auto num_rows = factors->get_size()[0];
    const auto factor_row_ptrs = factors->get_const_row_ptrs();
    const auto factor_cols = factors->get_const_col_idxs();
    const auto factor_vals = factors->get_values();
    array<IndexType> diag_idx_array{exec, num_rows};
    array<int> tmp_storage{exec};
    const auto diag_idxs = diag_idx_array.get_data();
    if (num_rows > 0) {
        syncfree_storage dep_storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::symbolic_factorize_simple(
            num_blocks, default_block_size, 0, exec->get_queue(), row_ptrs,
            col_idxs, factor_row_ptrs, factor_cols, lookup_offsets,
            lookup_storage, lookup_descs, diag_idxs, factor_vals, out_row_nnz,
            dep_storage, num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE);


template <typename IndexType>
void symbolic_factorize_simple_finalize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<float, IndexType>* factors, IndexType* out_col_idxs)
{
    auto policy = onedpl_policy(exec);
    const auto col_idxs = factors->get_const_col_idxs();
    const auto vals = factors->get_const_values();
    const auto input_it =
        oneapi::dpl::make_zip_iterator(as_device_type(vals), col_idxs);
    const auto output_it =
        dpl::make_zip_iterator(oneapi::dpl::discard_iterator(), out_col_idxs);
    std::copy_if(policy, input_it,
                 input_it + factors->get_num_stored_elements(), output_it,
                 [](auto tuple) { return std::get<0>(tuple) == one<float>(); });
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(
    GKO_DECLARE_LU_SYMMETRIC_FACTORIZE_SIMPLE_FINALIZE);


}  // namespace lu_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
