// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/multigrid/pgm_kernels.hpp"

#include <memory>

#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>

#include "common/cuda_hip/base/thrust.hpp"
#include "common/cuda_hip/base/types.hpp"
#include "common/cuda_hip/components/memory.hpp"
#include "common/cuda_hip/components/thread_ids.hpp"


namespace gko {
namespace kernels {
namespace GKO_DEVICE_NAMESPACE {
/**
 * @brief The PGM solver namespace.
 *
 * @ingroup pgm
 */
namespace pgm {
namespace kernels {


template <typename IndexType>
__global__ void match_edge(size_type size,
                           const IndexType* __restrict__ strongest_neighbor,
                           IndexType* __restrict__ agg)
{
    auto tidx = static_cast<IndexType>(thread::get_thread_id_flat<int64>());
    if (tidx >= size || load_relaxed(agg + tidx) != -1) {
        return;
    }
    auto neighbor = strongest_neighbor[tidx];
    if (neighbor != -1 && strongest_neighbor[neighbor] == tidx &&
        tidx <= neighbor) {
        store_relaxed(agg + tidx, tidx);
        store_relaxed(agg + neighbor, tidx);
    }
}


}  // namespace kernels


template <typename IndexType>
void match_edge(std::shared_ptr<const DefaultExecutor> exec,
                const array<IndexType>& strongest_neighbor,
                array<IndexType>& agg)
{
    constexpr int default_block_size = 512;
    auto num_blocks = ceildiv(agg.get_size(), default_block_size);
    kernels::
        match_edge<<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
            agg.get_size(), strongest_neighbor.get_const_data(),
            agg.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num,
              IndexType* row_idxs, IndexType* col_idxs)
{
    auto it = thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));
    thrust::sort(thrust_policy(exec), it, it + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec, size_type nnz,
                    IndexType* row_idxs, IndexType* col_idxs, ValueType* vals)
{
    auto vals_it = as_device_type(vals);
    auto it = thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));
    // Because reduce_by_key is not deterministic, so we do not need
    // stable_sort_by_key
    // TODO: If we have deterministic reduce_by_key, it should be
    // stable_sort_by_key
    thrust::sort_by_key(thrust_policy(exec), it, it + nnz, vals_it);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_SORT_ROW_MAJOR);


template <typename ValueType, typename IndexType>
void compute_coarse_coo(std::shared_ptr<const DefaultExecutor> exec,
                        size_type fine_nnz, const IndexType* row_idxs,
                        const IndexType* col_idxs, const ValueType* vals,
                        matrix::Coo<ValueType, IndexType>* coarse_coo)
{
    auto vals_it = as_device_type(vals);
    // this const_cast is necessary as a workaround for CCCL bug
    // https://github.com/NVIDIA/cccl/issues/1527
    // shipped with CUDA 12.4
    auto key_it = thrust::make_zip_iterator(thrust::make_tuple(
        const_cast<IndexType*>(row_idxs), const_cast<IndexType*>(col_idxs)));

    auto coarse_vals_it = as_device_type(coarse_coo->get_values());
    auto coarse_key_it = thrust::make_zip_iterator(thrust::make_tuple(
        coarse_coo->get_row_idxs(), coarse_coo->get_col_idxs()));

    thrust::reduce_by_key(thrust_policy(exec), key_it, key_it + fine_nnz,
                          vals_it, coarse_key_it, coarse_vals_it);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_COMPUTE_COARSE_COO);


}  // namespace pgm
}  // namespace GKO_DEVICE_NAMESPACE
}  // namespace kernels
}  // namespace gko
