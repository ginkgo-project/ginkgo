/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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


#include <memory>


#include <hip/hip_runtime.h>
#include <hipsparse.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/multigrid/amgx_pgm.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The AMGX_PGM solver namespace.
 *
 * @ingroup amgx_pgm
 */
namespace amgx_pgm {


constexpr int default_block_size = 512;


#include "common/multigrid/amgx_pgm_kernels.hpp.inc"


template <typename IndexType>
void match_edge(std::shared_ptr<const HipExecutor> exec,
                const Array<IndexType> &strongest_neighbor,
                Array<IndexType> &agg)
{
    const auto num = agg.get_num_elems();
    const dim3 grid(ceildiv(num, default_block_size));
    hipLaunchKernelGGL(kernel::match_edge_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0, num,
                       strongest_neighbor.get_const_data(), agg.get_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_MATCH_EDGE_KERNEL);


template <typename IndexType>
void count_unagg(std::shared_ptr<const HipExecutor> exec,
                 const Array<IndexType> &agg, IndexType *num_unagg)
{
    Array<IndexType> active_agg(exec, agg.get_num_elems());
    const dim3 grid(ceildiv(active_agg.get_num_elems(), default_block_size));
    hipLaunchKernelGGL(kernel::activate_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0,
                       active_agg.get_num_elems(), agg.get_const_data(),
                       active_agg.get_data());
    *num_unagg = reduce_add_array(exec, active_agg.get_num_elems(),
                                  active_agg.get_const_data());
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_COUNT_UNAGG_KERNEL);


template <typename IndexType>
void renumber(std::shared_ptr<const HipExecutor> exec, Array<IndexType> &agg,
              IndexType *num_agg)
{
    const auto num = agg.get_num_elems();
    Array<IndexType> agg_map(exec, num + 1);
    components::fill_array(exec, agg_map.get_data(), agg_map.get_num_elems(),
                           zero<IndexType>());
    const dim3 grid(ceildiv(num, default_block_size));
    hipLaunchKernelGGL(kernel::fill_agg_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0, num,
                       agg.get_const_data(), agg_map.get_data());
    components::prefix_sum(exec, agg_map.get_data(), agg_map.get_num_elems());
    hipLaunchKernelGGL(kernel::renumber_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0, num,
                       agg_map.get_const_data(), agg.get_data());
    *num_agg = exec->copy_val_to_host(agg_map.get_const_data() + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_RENUMBER_KERNEL);


template <typename ValueType, typename IndexType>
void find_strongest_neighbor(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *weight_mtx,
    const matrix::Diagonal<ValueType> *diag, Array<IndexType> &agg,
    Array<IndexType> &strongest_neighbor)
{
    const auto num = agg.get_num_elems();
    const dim3 grid(ceildiv(num, default_block_size));
    hipLaunchKernelGGL(
        kernel::find_strongest_neighbor_kernel, dim3(grid),
        dim3(default_block_size), 0, 0, num, weight_mtx->get_const_row_ptrs(),
        weight_mtx->get_const_col_idxs(), weight_mtx->get_const_values(),
        diag->get_const_values(), diag->get_stride(), agg.get_data(),
        strongest_neighbor.get_data());
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_FIND_STRONGEST_NEIGHBOR);


template <typename ValueType, typename IndexType>
void assign_to_exist_agg(std::shared_ptr<const HipExecutor> exec,
                         const matrix::Csr<ValueType, IndexType> *weight_mtx,
                         const matrix::Diagonal<ValueType> *diag,
                         Array<IndexType> &agg,
                         Array<IndexType> &intermediate_agg)
{
    auto agg_val = (intermediate_agg.get_num_elems() > 0)
                       ? intermediate_agg.get_data()
                       : agg.get_data();
    const auto num = agg.get_num_elems();
    const dim3 grid(ceildiv(num, default_block_size));
    hipLaunchKernelGGL(kernel::assign_to_exist_agg_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0, num,
                       weight_mtx->get_const_row_ptrs(),
                       weight_mtx->get_const_col_idxs(),
                       weight_mtx->get_const_values(), diag->get_const_values(),
                       diag->get_stride(), agg.get_const_data(), agg_val);
    if (intermediate_agg.get_num_elems() > 0) {
        // Copy the intermediate_agg to agg
        agg = intermediate_agg;
    }
}

GKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_AMGX_PGM_ASSIGN_TO_EXIST_AGG);


template <typename ValueType, typename IndexType>
void amgx_pgm_generate(std::shared_ptr<const HipExecutor> exec,
                       const matrix::Csr<ValueType, IndexType> *source,
                       const Array<IndexType> &agg,
                       matrix::Csr<ValueType, IndexType> *coarse,
                       matrix::Csr<ValueType, IndexType> *temp)
{
    const auto source_nrows = source->get_size()[0];
    const auto source_nnz = source->get_num_stored_elements();
    const auto coarse_nrows = coarse->get_size()[0];
    Array<IndexType> row_map(exec, source_nrows);
    // fill coarse row pointer as zero
    components::fill_array(exec, temp->get_row_ptrs(), coarse_nrows + 1,
                           zero<IndexType>());
    // compute each source row should be moved and also change column index
    dim3 grid(ceildiv(source_nrows, default_block_size));
    // agg source_row (for row size) coarse row source map
    hipLaunchKernelGGL(kernel::get_source_row_map_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0, source_nrows,
                       agg.get_const_data(), source->get_const_row_ptrs(),
                       temp->get_row_ptrs(), row_map.get_data());
    // prefix sum of temp_row_ptrs
    components::prefix_sum(exec, temp->get_row_ptrs(), coarse_nrows + 1);
    // copy source -> to coarse and change column index
    hipLaunchKernelGGL(
        kernel::move_row_kernel, dim3(grid), dim3(default_block_size), 0, 0,
        source_nrows, agg.get_const_data(), row_map.get_const_data(),
        source->get_const_row_ptrs(), source->get_const_col_idxs(),
        as_hip_type(source->get_const_values()), temp->get_const_row_ptrs(),
        temp->get_col_idxs(), as_hip_type(temp->get_values()));
    // sort csr
    csr::sort_by_column_index(exec, temp);
    // summation of the elements with same position
    grid = ceildiv(coarse_nrows, default_block_size);
    hipLaunchKernelGGL(kernel::merge_col_kernel, dim3(grid),
                       dim3(default_block_size), 0, 0, coarse_nrows,
                       temp->get_const_row_ptrs(), temp->get_col_idxs(),
                       as_hip_type(temp->get_values()), coarse->get_row_ptrs());
    // build the coarse matrix
    components::prefix_sum(exec, coarse->get_row_ptrs(), coarse_nrows + 1);
    // prefix sum of coarse->get_row_ptrs
    const auto coarse_nnz =
        exec->copy_val_to_host(coarse->get_row_ptrs() + coarse_nrows);
    // reallocate size of column and values
    matrix::CsrBuilder<ValueType, IndexType> coarse_builder{coarse};
    auto &coarse_col_idxs_array = coarse_builder.get_col_idx_array();
    auto &coarse_vals_array = coarse_builder.get_value_array();
    coarse_col_idxs_array.resize_and_reset(coarse_nnz);
    coarse_vals_array.resize_and_reset(coarse_nnz);
    // copy the result
    hipLaunchKernelGGL(
        kernel::copy_to_coarse_kernel, dim3(grid), dim3(default_block_size), 0,
        0, coarse_nrows, temp->get_const_row_ptrs(), temp->get_const_col_idxs(),
        as_hip_type(temp->get_const_values()), coarse->get_const_row_ptrs(),
        coarse_col_idxs_array.get_data(),
        as_hip_type(coarse_vals_array.get_data()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_AMGX_PGM_GENERATE);


}  // namespace amgx_pgm
}  // namespace hip
}  // namespace kernels
}  // namespace gko
