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

#include "core/constraints/constraints_handler_kernels.hpp"


#include <memory>


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>


#include "hip/base/config.hip.hpp"
#include "hip/base/pointer_mode_guard.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace cons {

constexpr int default_block_size = 512;


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void fill_subset(
    size_type n, const IndexType* __restrict__ map,
    ValueType* __restrict__ array, ValueType val)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < n) {
        array[map[tidx]] = val;
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void copy_subset(
    size_type n, const IndexType* __restrict__ map,
    const ValueType* __restrict__ src, ValueType* __restrict__ dst)
{
    const auto tidx = thread::get_thread_id_flat();
    if (tidx < n) {
        const auto mapped_idx = map[tidx];
        dst[mapped_idx] = src[mapped_idx];
    }
}


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void set_unit_rows(
    size_type n, const IndexType* __restrict__ map,
    const IndexType* __restrict__ row_ptrs,
    const IndexType* __restrict__ col_idxs, ValueType* __restrict__ values)
{
    constexpr auto warp_size = config::warp_size;
    const auto subset_idx = thread::get_subwarp_id_flat<warp_size>();
    const auto local_tidx = threadIdx.x % warp_size;

    if (subset_idx < n) {
        const auto row = map[subset_idx];
        for (size_type i = local_tidx; i < row_ptrs[row + 1] - row_ptrs[row];
             i += warp_size) {
            const auto orig_idx = i + row_ptrs[row];
            values[orig_idx] = col_idxs[orig_idx] == row;
        }
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void fill_subset(std::shared_ptr<const DefaultExecutor> exec,
                 const Array<IndexType>& subset, ValueType* data, ValueType val)
{
    const auto n = subset.get_num_elems();
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
    hipLaunchKernelGGL(kernel::fill_subset, dim3(grid_size), dim3(block_size),
                       0, 0, n, as_hip_type(subset.get_const_data()),
                       as_hip_type(data), as_hip_type(val));
}


template <typename ValueType, typename IndexType>
void copy_subset(std::shared_ptr<const DefaultExecutor> exec,
                 const Array<IndexType>& subset, const ValueType* src,
                 ValueType* dst)
{
    const auto n = subset.get_num_elems();
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
    hipLaunchKernelGGL(kernel::copy_subset, dim3(grid_size), dim3(block_size),
                       0, 0, n, as_hip_type(subset.get_const_data()),
                       as_hip_type(src), as_hip_type(dst));
}


template <typename ValueType, typename IndexType>
void set_unit_rows(std::shared_ptr<const DefaultExecutor> exec,
                   const Array<IndexType>& subset, const IndexType* row_ptrs,
                   const IndexType* col_idxs, ValueType* values)
{
    const auto subset_rows = subset.get_num_elems();
    const auto num_blocks =
        ceildiv(config::warp_size * subset_rows, default_block_size);

    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::set_unit_rows), dim3(num_blocks),
                       dim3(default_block_size), 0, 0, subset_rows,
                       as_hip_type(subset.get_const_data()),
                       as_hip_type(row_ptrs), as_hip_type(col_idxs),
                       as_hip_type(values));
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONS_FILL_SUBSET);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONS_COPY_SUBSET);
GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_CONS_SET_UNIT_ROWS);


}  // namespace cons
}  // namespace hip
}  // namespace kernels
}  // namespace gko
