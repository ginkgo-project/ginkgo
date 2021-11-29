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

#include "core/multigrid/amgx_pgm_kernels.hpp"


// #include <dpcpp/base/math.hpp>
// #include <dpct/blas_utils.hpp>
// #include <dpct/dpl_utils.hpp>
#include <memory>


#include <CL/sycl.hpp>
#include <oneapi/mkl.hpp>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/base/onemkl_bindings.hpp"
#include "dpcpp/components/atomic.dp.hpp"
#include "dpcpp/components/reduction.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The AMGX_PGM solver namespace.
 *
 * @ingroup amgx_pgm
 */
namespace amgx_pgm {


constexpr int default_block_size = 256;


namespace kernel {


template <typename IndexType>
void match_edge_kernel(size_type num,
                       const IndexType* __restrict__ strongest_neighbor_vals,
                       IndexType* __restrict__ agg_vals,
                       sycl::nd_item<3> item_ct1)
{
    auto tidx = thread::get_thread_id_flat<IndexType>(item_ct1);
    if (tidx >= num) {
        return;
    }
    if (agg_vals[tidx] != -1) {
        return;
    }
    auto neighbor = strongest_neighbor_vals[tidx];
    if (neighbor != -1 && strongest_neighbor_vals[neighbor] == tidx &&
        tidx <= neighbor) {
        // Use the smaller index as agg point
        agg_vals[tidx] = tidx;
        agg_vals[neighbor] = tidx;
    }
}

template <typename IndexType>
void match_edge_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                       sycl::queue* queue, size_type num,
                       const IndexType* strongest_neighbor_vals,
                       IndexType* agg_vals)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            match_edge_kernel(num, strongest_neighbor_vals, agg_vals, item_ct1);
        });
}


template <typename IndexType>
void activate_kernel(size_type num, const IndexType* __restrict__ agg,
                     IndexType* __restrict__ active_agg,
                     sycl::nd_item<3> item_ct1)
{
    auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx >= num) {
        return;
    }
    active_agg[tidx] = agg[tidx] == -1;
}

template <typename IndexType>
void activate_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, size_type num, const IndexType* agg,
                     IndexType* active_agg)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            activate_kernel(num, agg, active_agg, item_ct1);
                        });
}


template <typename IndexType>
void fill_agg_kernel(size_type num, const IndexType* __restrict__ index,
                     IndexType* __restrict__ result, sycl::nd_item<3> item_ct1)
{
    auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx >= num) {
        return;
    }
    // agg_vals[i] == i always holds in the aggregated group whose identifier is
    // i because we use the index of element as the aggregated group identifier.
    result[tidx] = (index[tidx] == tidx);
}

template <typename IndexType>
void fill_agg_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, size_type num, const IndexType* index,
                     IndexType* result)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            fill_agg_kernel(num, index, result, item_ct1);
                        });
}


template <typename IndexType>
void renumber_kernel(size_type num, const IndexType* __restrict__ map,
                     IndexType* __restrict__ result, sycl::nd_item<3> item_ct1)
{
    auto tidx = thread::get_thread_id_flat(item_ct1);
    if (tidx >= num) {
        return;
    }
    result[tidx] = map[result[tidx]];
}

template <typename IndexType>
void renumber_kernel(dim3 grid, dim3 block, size_type dynamic_shared_memory,
                     sycl::queue* queue, size_type num, const IndexType* map,
                     IndexType* result)
{
    queue->parallel_for(sycl_nd_range(grid, block),
                        [=](sycl::nd_item<3> item_ct1) {
                            renumber_kernel(num, map, result, item_ct1);
                        });
}


template <typename ValueType, typename IndexType>
void find_strongest_neighbor_kernel(const size_type num,
                                    const IndexType* __restrict__ row_ptrs,
                                    const IndexType* __restrict__ col_idxs,
                                    const ValueType* __restrict__ weight_vals,
                                    const ValueType* __restrict__ diag,
                                    IndexType* __restrict__ agg,
                                    IndexType* __restrict__ strongest_neighbor,
                                    sycl::nd_item<3> item_ct1)
{
    auto row = thread::get_thread_id_flat(item_ct1);
    if (row >= num) {
        return;
    }

    auto max_weight_unagg = zero<ValueType>();
    auto max_weight_agg = zero<ValueType>();
    IndexType strongest_unagg = -1;
    IndexType strongest_agg = -1;
    if (agg[row] != -1) {
        return;
    }
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight =
            weight_vals[idx] / max(std::abs(diag[row]), std::abs(diag[col]));
        if (agg[col] == -1 &&
            (weight > max_weight_unagg ||
             (weight == max_weight_unagg && col > strongest_unagg))) {
            max_weight_unagg = weight;
            strongest_unagg = col;
        } else if (agg[col] != -1 &&
                   (weight > max_weight_agg ||
                    (weight == max_weight_agg && col > strongest_agg))) {
            max_weight_agg = weight;
            strongest_agg = col;
        }
    }

    if (strongest_unagg == -1 && strongest_agg != -1) {
        // all neighbor is agg, connect to the strongest agg
        // Also, no others will use this item as their strongest_neighbor
        // because they are already aggregated. Thus, it is determinstic
        // behavior
        agg[row] = agg[strongest_agg];
    } else if (strongest_unagg != -1) {
        // set the strongest neighbor in the unagg group
        strongest_neighbor[row] = strongest_unagg;
    } else {
        // no neighbor
        strongest_neighbor[row] = row;
    }
}

template <typename ValueType, typename IndexType>
void find_strongest_neighbor_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const size_type num, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* weight_vals, const ValueType* diag, IndexType* agg,
    IndexType* strongest_neighbor)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            find_strongest_neighbor_kernel(num, row_ptrs, col_idxs, weight_vals,
                                           diag, agg, strongest_neighbor,
                                           item_ct1);
        });
}


template <typename ValueType, typename IndexType>
void assign_to_exist_agg_kernel(const size_type num,
                                const IndexType* __restrict__ row_ptrs,
                                const IndexType* __restrict__ col_idxs,
                                const ValueType* __restrict__ weight_vals,
                                const ValueType* __restrict__ diag,
                                const IndexType* __restrict__ agg_const_val,
                                IndexType* __restrict__ agg_val,
                                sycl::nd_item<3> item_ct1)
{
    auto row = thread::get_thread_id_flat(item_ct1);
    if (row >= num || agg_val[row] != -1) {
        return;
    }
    ValueType max_weight_agg = zero<ValueType>();
    IndexType strongest_agg = -1;
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight =
            weight_vals[idx] / max(std::abs(diag[row]), std::abs(diag[col]));
        if (agg_const_val[col] != -1 &&
            (weight > max_weight_agg ||
             (weight == max_weight_agg && col > strongest_agg))) {
            max_weight_agg = weight;
            strongest_agg = col;
        }
    }
    if (strongest_agg != -1) {
        agg_val[row] = agg_const_val[strongest_agg];
    } else {
        agg_val[row] = row;
    }
}

template <typename ValueType, typename IndexType>
void assign_to_exist_agg_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const size_type num, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* weight_vals, const ValueType* diag,
    const IndexType* agg_const_val, IndexType* agg_val)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            assign_to_exist_agg_kernel(num, row_ptrs, col_idxs, weight_vals,
                                       diag, agg_const_val, agg_val, item_ct1);
        });
}

// This is the undeterminstic implementation which is the same implementation of
// the previous one but agg_val == agg_const_val.
template <typename ValueType, typename IndexType>
void assign_to_exist_agg_kernel(const size_type num,
                                const IndexType* __restrict__ row_ptrs,
                                const IndexType* __restrict__ col_idxs,
                                const ValueType* __restrict__ weight_vals,
                                const ValueType* __restrict__ diag,
                                IndexType* __restrict__ agg_val,
                                sycl::nd_item<3> item_ct1)
{
    auto row = thread::get_thread_id_flat(item_ct1);
    if (row >= num || agg_val[row] != -1) {
        return;
    }
    ValueType max_weight_agg = zero<ValueType>();
    IndexType strongest_agg = -1;
    for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; idx++) {
        auto col = col_idxs[idx];
        if (col == row) {
            continue;
        }
        auto weight =
            weight_vals[idx] / max(std::abs(diag[row]), std::abs(diag[col]));
        if (agg_val[col] != -1 &&
            (weight > max_weight_agg ||
             (weight == max_weight_agg && col > strongest_agg))) {
            max_weight_agg = weight;
            strongest_agg = col;
        }
    }
    if (strongest_agg != -1) {
        agg_val[row] = agg_val[strongest_agg];
    } else {
        agg_val[row] = row;
    }
}

template <typename ValueType, typename IndexType>
void assign_to_exist_agg_kernel(
    dim3 grid, dim3 block, size_type dynamic_shared_memory, sycl::queue* queue,
    const size_type num, const IndexType* row_ptrs, const IndexType* col_idxs,
    const ValueType* weight_vals, const ValueType* diag, IndexType* agg_val)
{
    queue->parallel_for(
        sycl_nd_range(grid, block), [=](sycl::nd_item<3> item_ct1) {
            assign_to_exist_agg_kernel(num, row_ptrs, col_idxs, weight_vals,
                                       diag, agg_val, item_ct1);
        });
}


}  // namespace kernel


template <typename IndexType>
void match_edge(std::shared_ptr<const DpcppExecutor> exec,
                const Array<IndexType>& strongest_neighbor,
                Array<IndexType>& agg)
{
    const auto num = agg.get_num_elems();
    const dim3 grid(ceildiv(num, default_block_size));
    kernel::match_edge_kernel(grid, default_block_size, 0, exec->get_queue(),
                              num, strongest_neighbor.get_const_data(),
                              agg.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const DpcppExecutor> exec,
                 const Array<IndexType>& agg, IndexType* num_unagg)
{
    Array<IndexType> active_agg(exec, agg.get_num_elems());
    const dim3 grid(ceildiv(active_agg.get_num_elems(), default_block_size));
    kernel::activate_kernel(grid, default_block_size, 0, exec->get_queue(),
                            active_agg.get_num_elems(), agg.get_const_data(),
                            active_agg.get_data());
    *num_unagg = reduce_add_array(exec, active_agg.get_num_elems(),
                                  active_agg.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const DpcppExecutor> exec, Array<IndexType>& agg,
              IndexType* num_agg)
{
    const auto num = agg.get_num_elems();
    Array<IndexType> agg_map(exec, num + 1);
    const dim3 grid(ceildiv(num, default_block_size));
    kernel::fill_agg_kernel(grid, default_block_size, 0, exec->get_queue(), num,
                            agg.get_const_data(), agg_map.get_data());
    components::prefix_sum(exec, agg_map.get_data(), agg_map.get_num_elems());
    kernel::renumber_kernel(grid, default_block_size, 0, exec->get_queue(), num,
                            agg_map.get_const_data(), agg.get_data());
    *num_agg = exec->copy_val_to_host(agg_map.get_const_data() + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Csr<ValueType, IndexType>* weight_mtx,
    const matrix::Diagonal<ValueType>* diag, Array<IndexType>& agg,
    Array<IndexType>& strongest_neighbor)
{
    const auto num = agg.get_num_elems();
    const dim3 grid(ceildiv(num, default_block_size));
    kernel::find_strongest_neighbor_kernel(
        grid, default_block_size, 0, exec->get_queue(), num,
        weight_mtx->get_const_row_ptrs(), weight_mtx->get_const_col_idxs(),
        weight_mtx->get_const_values(), diag->get_const_values(),
        agg.get_data(), strongest_neighbor.get_data());
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_FIND_STRONGEST_NEIGHBOR);

template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const DpcppExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* weight_mtx,
                         const matrix::Diagonal<ValueType>* diag,
                         Array<IndexType>& agg,
                         Array<IndexType>& intermediate_agg)
{
    const auto num = agg.get_num_elems();
    const dim3 grid(ceildiv(num, default_block_size));
    if (intermediate_agg.get_num_elems() > 0) {
        // determinstic kernel
        kernel::assign_to_exist_agg_kernel(
            grid, default_block_size, 0, exec->get_queue(), num,
            weight_mtx->get_const_row_ptrs(), weight_mtx->get_const_col_idxs(),
            weight_mtx->get_const_values(), diag->get_const_values(),
            agg.get_const_data(), intermediate_agg.get_data());
        // Copy the intermediate_agg to agg
        agg = intermediate_agg;
    } else {
        // undeterminstic kernel
        kernel::assign_to_exist_agg_kernel(
            grid, default_block_size, 0, exec->get_queue(), num,
            weight_mtx->get_const_row_ptrs(), weight_mtx->get_const_col_idxs(),
            weight_mtx->get_const_values(), diag->get_const_values(),
            agg.get_data());
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_ASSIGN_TO_EXIST_AGG);


}  // namespace amgx_pgm
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
