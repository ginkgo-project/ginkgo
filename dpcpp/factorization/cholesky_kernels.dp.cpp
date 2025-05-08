// SPDX-FileCopyrightText: 2017 - 2025 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include <oneapi/dpl/algorithm>

#include "core/factorization/cholesky_kernels.hpp"

#include <algorithm>
#include <memory>

#include <sycl/sycl.hpp>

#include <ginkgo/core/matrix/csr.hpp>

#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/onedpl.hpp"
#include "dpcpp/components/syncfree.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The Cholesky namespace.
 *
 * @ingroup factor
 */
namespace cholesky {


constexpr int default_block_size = 512;


namespace kernel {


template <bool full_fillin, typename ValueType, typename IndexType>
void factorize(
    const IndexType* __restrict__ row_ptrs, const IndexType* __restrict__ cols,
    const IndexType* __restrict__ storage_offsets,
    const int32* __restrict__ storage, const int64* __restrict__ row_descs,
    const IndexType* __restrict__ diag_idxs,
    const IndexType* __restrict__ transpose_idxs, ValueType* __restrict__ vals,
    syncfree_storage dep_storage, size_type num_rows, sycl::nd_item<3> item_ct1,
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
    const auto row_begin = row_ptrs[row];
    const auto row_diag = diag_idxs[row];
    const auto row_end = row_ptrs[row + 1];
    gko::matrix::csr::device_sparsity_lookup<IndexType> lookup{
        row_ptrs, cols,      storage_offsets,
        storage,  row_descs, static_cast<size_type>(row)};
    // for each lower triangular entry: eliminate with corresponding column
    for (auto lower_nz = row_begin; lower_nz < row_diag; lower_nz++) {
        const auto dep = cols[lower_nz];
        scheduler.wait(dep);
        const auto scale = vals[lower_nz];
        const auto diag_idx = diag_idxs[dep];
        const auto dep_end = row_ptrs[dep + 1];
        // subtract column dep from current column
        for (auto upper_nz = diag_idx + lane; upper_nz < dep_end;
             upper_nz += config::warp_size) {
            const auto upper_col = cols[upper_nz];
            if (upper_col >= row) {
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
    }
    auto diag_val = sqrt(vals[row_diag]);
    for (auto upper_nz = row_diag + 1 + lane; upper_nz < row_end;
         upper_nz += config::warp_size) {
        vals[upper_nz] /= diag_val;
        // copy the upper triangular entries to the transpose
        vals[transpose_idxs[upper_nz]] = conj(vals[upper_nz]);
    }
    if (lane == 0) {
        // store computed diagonal
        vals[row_diag] = diag_val;
    }
    scheduler.mark_ready();
}

template <bool full_fillin, typename ValueType, typename IndexType>
void factorize(dim3 grid, dim3 block, gko::size_type, sycl::queue* queue,
               const IndexType* row_ptrs, const IndexType* cols,
               const IndexType* storage_offsets, const int32* storage,
               const int64* row_descs, const IndexType* diag_idxs,
               const IndexType* transpose_idxs, ValueType* vals,
               syncfree_storage dep_storage, size_type num_rows)
{
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
                                     row_descs, diag_idxs, transpose_idxs, vals,
                                     dep_storage, num_rows, item_ct1,
                                     *sh_dep_storage_acc_ct1.get_pointer());
                             });
    });
};


}  // namespace kernel


template <typename ValueType, typename IndexType>
void symbolic_count(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* mtx,
                    const factorization::elimination_forest<IndexType>& forest,
                    IndexType* row_nnz, array<IndexType>& tmp_storage)
{
    const auto num_rows = mtx->get_size()[0];
    const auto mtx_nnz = mtx->get_num_stored_elements();
    tmp_storage.resize_and_reset(mtx_nnz + num_rows);
    const auto postorder_cols = tmp_storage.get_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto cols = mtx->get_const_col_idxs();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    auto queue = exec->get_queue();
    // build sorted postorder node list for each row
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx_id) {
            const auto row = idx_id[0];
            const auto row_begin = row_ptrs[row];
            const auto row_end = row_ptrs[row + 1];
            auto lower_end = row_begin;
            for (auto nz = row_begin; nz < row_end; nz++) {
                const auto col = cols[nz];
                if (col < row) {
                    postorder_cols[lower_end] = inv_postorder[cols[nz]];
                    lower_end++;
                }
            }
            // heap-sort the elements
            std::make_heap(postorder_cols + row_begin,
                           postorder_cols + lower_end);
            std::sort_heap(postorder_cols + row_begin,
                           postorder_cols + lower_end);
            lower_ends[row] = lower_end;
        });
    });
    // count nonzeros per row of L
    queue->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx_id) {
            const auto row = idx_id[0];
            const auto row_begin = row_ptrs[row];
            // instead of relying on the input containing a diagonal, we
            // artificially introduce the diagonal entry (in postorder indexing)
            // as a sentinel after the last lower triangular entry.
            const auto diag_postorder = inv_postorder[row];
            const auto lower_end = lower_ends[row];
            IndexType count{};
            for (auto nz = row_begin; nz < lower_end; ++nz) {
                auto node = postorder_cols[nz];
                const auto next_node = nz < lower_end - 1
                                           ? postorder_cols[nz + 1]
                                           : diag_postorder;
                while (node < next_node) {
                    count++;
                    node = postorder_parent[node];
                }
            }
            row_nnz[row] = count + 1;  // lower entries plus diagonal
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT);


template <typename ValueType, typename IndexType>
void symbolic_factorize(
    std::shared_ptr<const DefaultExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* mtx,
    const factorization::elimination_forest<IndexType>& forest,
    matrix::Csr<ValueType, IndexType>* l_factor,
    const array<IndexType>& tmp_storage)
{
    const auto num_rows = mtx->get_size()[0];
    const auto mtx_nnz = mtx->get_num_stored_elements();
    const auto postorder_cols = tmp_storage.get_const_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto postorder = forest.postorder.get_const_data();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    const auto out_row_ptrs = l_factor->get_const_row_ptrs();
    const auto out_cols = l_factor->get_col_idxs();
    exec->get_queue()->submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>{num_rows}, [=](sycl::id<1> idx_id) {
            const auto row = idx_id[0];
            const auto row_begin = row_ptrs[row];
            // instead of relying on the input containing a diagonal, we
            // artificially introduce the diagonal entry (in postorder indexing)
            // as a sentinel after the last lower triangular entry.
            const auto diag_postorder = inv_postorder[row];
            const auto lower_end = lower_ends[row];
            auto out_nz = out_row_ptrs[row];
            for (auto nz = row_begin; nz < lower_end; ++nz) {
                auto node = postorder_cols[nz];
                const auto next_node = nz < lower_end - 1
                                           ? postorder_cols[nz + 1]
                                           : diag_postorder;
                while (node < next_node) {
                    out_cols[out_nz] = postorder[node];
                    out_nz++;
                    node = postorder_parent[node];
                }
            }
            // add diagonal entry
            out_cols[out_nz] = row;
        });
    });
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_FACTORIZE);


template <typename ValueType, typename IndexType>
void initialize(std::shared_ptr<const DefaultExecutor> exec,
                const matrix::Csr<ValueType, IndexType>* mtx,
                const IndexType* factor_lookup_offsets,
                const int64* factor_lookup_descs,
                const int32* factor_lookup_storage, IndexType* diag_idxs,
                IndexType* transpose_idxs,
                matrix::Csr<ValueType, IndexType>* factors)
{
    lu_factorization::initialize(exec, mtx, factor_lookup_offsets,
                                 factor_lookup_descs, factor_lookup_storage,
                                 diag_idxs, factors);
    // convert to COO
    const auto nnz = factors->get_num_stored_elements();
    array<IndexType> row_idx_array{exec, nnz};
    array<IndexType> col_idx_array{exec, nnz};
    const auto row_idxs = row_idx_array.get_data();
    const auto col_idxs = col_idx_array.get_data();
    exec->copy(nnz, factors->get_const_col_idxs(), col_idxs);
    components::convert_ptrs_to_idxs(exec, factors->get_const_row_ptrs(),
                                     factors->get_size()[0], row_idxs);
    components::fill_seq_array(exec, transpose_idxs, nnz);
    // compute nonzero permutation for sparse transpose
    // TODO: check sort performance on both or twice
    // dpl 2022.7.0 introduces stable_sort_by_key
    oneapi::dpl::sort_by_key(onedpl_policy(exec), row_idxs, row_idxs + nnz,
                             transpose_idxs);
    oneapi::dpl::sort_by_key(onedpl_policy(exec), col_idxs, col_idxs + nnz,
                             transpose_idxs);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_INITIALIZE);


template <typename ValueType, typename IndexType>
void factorize(std::shared_ptr<const DefaultExecutor> exec,
               const IndexType* lookup_offsets, const int64* lookup_descs,
               const int32* lookup_storage, const IndexType* diag_idxs,
               const IndexType* transpose_idxs,
               const factorization::elimination_forest<IndexType>& forest,
               matrix::Csr<ValueType, IndexType>* factors, bool full_fillin,
               array<int>& tmp_storage)
{
    const auto num_rows = factors->get_size()[0];
    if (num_rows > 0) {
        syncfree_storage storage(exec, tmp_storage, num_rows);
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        if (!full_fillin) {
            kernel::factorize<false>(
                num_blocks, default_block_size, 0, exec->get_queue(),
                factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
                lookup_offsets, lookup_storage, lookup_descs, diag_idxs,
                transpose_idxs, as_device_type(factors->get_values()), storage,
                num_rows);
        } else {
            kernel::factorize<true>(
                num_blocks, default_block_size, 0, exec->get_queue(),
                factors->get_const_row_ptrs(), factors->get_const_col_idxs(),
                lookup_offsets, lookup_storage, lookup_descs, diag_idxs,
                transpose_idxs, as_device_type(factors->get_values()), storage,
                num_rows);
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CHOLESKY_FACTORIZE);


}  // namespace cholesky
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
