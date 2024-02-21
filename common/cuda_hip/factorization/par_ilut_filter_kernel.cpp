// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "common/cuda_hip/base/config.hpp"
#include "common/cuda_hip/base/math.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/cooperative_groups.hpp"
#include "common/cuda_hip/components/intrinsics.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"
#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr int default_block_size = 512;


// subwarp sizes for filter kernels
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


namespace kernel {


template <int subwarp_size, typename IndexType, typename Predicate,
          typename BeginCallback, typename StepCallback,
          typename FinishCallback>
__device__ void abstract_filter_impl(const IndexType* row_ptrs,
                                     IndexType num_rows, Predicate pred,
                                     BeginCallback begin_cb,
                                     StepCallback step_cb,
                                     FinishCallback finish_cb)
{
    auto subwarp =
        group::tiled_partition<subwarp_size>(group::this_thread_block());
    auto row = thread::get_subwarp_id_flat<subwarp_size, IndexType>();
    auto lane = subwarp.thread_rank();
    auto lane_prefix_mask = (config::lane_mask_type(1) << lane) - 1;
    if (row >= num_rows) {
        return;
    }

    auto begin = row_ptrs[row];
    auto end = row_ptrs[row + 1];
    begin_cb(row);
    auto num_steps = ceildiv(end - begin, subwarp_size);
    for (IndexType step = 0; step < num_steps; ++step) {
        auto idx = begin + lane + step * subwarp_size;
        auto keep = idx < end && pred(idx, begin, end);
        auto mask = subwarp.ballot(keep);
        step_cb(row, idx, keep, popcnt(mask), popcnt(mask & lane_prefix_mask));
    }
    finish_cb(row, lane);
}


template <int subwarp_size, typename Predicate, typename IndexType>
__device__ void abstract_filter_nnz(const IndexType* __restrict__ row_ptrs,
                                    IndexType num_rows, Predicate pred,
                                    IndexType* __restrict__ nnz)
{
    IndexType count{};
    abstract_filter_impl<subwarp_size>(
        row_ptrs, num_rows, pred, [&](IndexType) { count = 0; },
        [&](IndexType, IndexType, bool, IndexType warp_count, IndexType) {
            count += warp_count;
        },
        [&](IndexType row, IndexType lane) {
            if (row < num_rows && lane == 0) {
                nnz[row] = count;
            }
        });
}


template <int subwarp_size, typename Predicate, typename IndexType,
          typename ValueType>
__device__ void abstract_filter(const IndexType* __restrict__ old_row_ptrs,
                                const IndexType* __restrict__ old_col_idxs,
                                const ValueType* __restrict__ old_vals,
                                IndexType num_rows, Predicate pred,
                                const IndexType* __restrict__ new_row_ptrs,
                                IndexType* __restrict__ new_row_idxs,
                                IndexType* __restrict__ new_col_idxs,
                                ValueType* __restrict__ new_vals)
{
    IndexType count{};
    IndexType new_offset{};
    abstract_filter_impl<subwarp_size>(
        old_row_ptrs, num_rows, pred,
        [&](IndexType row) {
            new_offset = new_row_ptrs[row];
            count = 0;
        },
        [&](IndexType row, IndexType idx, bool keep, IndexType warp_count,
            IndexType warp_prefix_sum) {
            if (keep) {
                auto new_idx = new_offset + warp_prefix_sum + count;
                if (new_row_idxs) {
                    new_row_idxs[new_idx] = row;
                }
                new_col_idxs[new_idx] = old_col_idxs[idx];
                new_vals[new_idx] = old_vals[idx];
            }
            count += warp_count;
        },
        [](IndexType, IndexType) {});
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void threshold_filter_nnz(
    const IndexType* __restrict__ row_ptrs, const ValueType* vals,
    IndexType num_rows, remove_complex<ValueType> threshold,
    IndexType* __restrict__ nnz, bool lower)
{
    abstract_filter_nnz<subwarp_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return abs(vals[idx]) >= threshold || idx == diag_idx;
        },
        nnz);
}


template <int subwarp_size, typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void threshold_filter(
    const IndexType* __restrict__ old_row_ptrs,
    const IndexType* __restrict__ old_col_idxs,
    const ValueType* __restrict__ old_vals, IndexType num_rows,
    remove_complex<ValueType> threshold,
    const IndexType* __restrict__ new_row_ptrs,
    IndexType* __restrict__ new_row_idxs, IndexType* __restrict__ new_col_idxs,
    ValueType* __restrict__ new_vals, bool lower)
{
    abstract_filter<subwarp_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            auto diag_idx = lower ? row_end - 1 : row_begin;
            return abs(old_vals[idx]) >= threshold || idx == diag_idx;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals);
}


template <int subwarp_size, typename IndexType, typename BucketType>
__global__ __launch_bounds__(default_block_size) void bucket_filter_nnz(
    const IndexType* __restrict__ row_ptrs, const BucketType* buckets,
    IndexType num_rows, BucketType bucket, IndexType* __restrict__ nnz)
{
    abstract_filter_nnz<subwarp_size>(
        row_ptrs, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        nnz);
}


template <int subwarp_size, typename ValueType, typename IndexType,
          typename BucketType>
__global__ __launch_bounds__(default_block_size) void bucket_filter(
    const IndexType* __restrict__ old_row_ptrs,
    const IndexType* __restrict__ old_col_idxs,
    const ValueType* __restrict__ old_vals, const BucketType* buckets,
    IndexType num_rows, BucketType bucket,
    const IndexType* __restrict__ new_row_ptrs,
    IndexType* __restrict__ new_row_idxs, IndexType* __restrict__ new_col_idxs,
    ValueType* __restrict__ new_vals)
{
    abstract_filter<subwarp_size>(
        old_row_ptrs, old_col_idxs, old_vals, num_rows,
        [&](IndexType idx, IndexType row_begin, IndexType row_end) {
            return buckets[idx] >= bucket || idx == row_end - 1;
        },
        new_row_ptrs, new_row_idxs, new_col_idxs, new_vals);
}


}  // namespace kernel


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter(syn::value_list<int, subwarp_size>,
                      std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType>* m_out,
                      matrix::Coo<ValueType, IndexType>* m_out_coo, bool lower)
{
    auto old_row_ptrs = a->get_const_row_ptrs();
    auto old_col_idxs = a->get_const_col_idxs();
    auto old_vals = a->get_const_values();
    // compute nnz for each row
    auto num_rows = static_cast<IndexType>(a->get_size()[0]);
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, block_size);
    auto new_row_ptrs = m_out->get_row_ptrs();
    if (num_blocks > 0) {
        kernel::threshold_filter_nnz<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                old_row_ptrs, as_device_type(old_vals), num_rows,
                as_device_type(threshold), new_row_ptrs, lower);
    }

    // build row pointers
    components::prefix_sum_nonnegative(exec, new_row_ptrs, num_rows + 1);

    // build matrix
    auto new_nnz = exec->copy_val_to_host(new_row_ptrs + num_rows);
    // resize arrays and update aliases
    matrix::CsrBuilder<ValueType, IndexType> builder{m_out};
    builder.get_col_idx_array().resize_and_reset(new_nnz);
    builder.get_value_array().resize_and_reset(new_nnz);
    auto new_col_idxs = m_out->get_col_idxs();
    auto new_vals = m_out->get_values();
    IndexType* new_row_idxs{};
    if (m_out_coo) {
        matrix::CooBuilder<ValueType, IndexType> coo_builder{m_out_coo};
        coo_builder.get_row_idx_array().resize_and_reset(new_nnz);
        coo_builder.get_col_idx_array() =
            make_array_view(exec, new_nnz, new_col_idxs);
        coo_builder.get_value_array() =
            make_array_view(exec, new_nnz, new_vals);
        new_row_idxs = m_out_coo->get_row_idxs();
    }
    if (num_blocks > 0) {
        kernel::threshold_filter<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                old_row_ptrs, old_col_idxs, as_device_type(old_vals), num_rows,
                as_device_type(threshold), new_row_ptrs, new_row_idxs,
                new_col_idxs, as_device_type(new_vals), lower);
    }
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_threshold_filter, threshold_filter);


}  // namespace

template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType>* m_out,
                      matrix::Coo<ValueType, IndexType>* m_out_coo, bool lower)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz = a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_threshold_filter(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, threshold, m_out,
        m_out_coo, lower);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
