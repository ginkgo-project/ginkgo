// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CUDA_COMPONENTS_REDUCTION_CUH_
#define GKO_CUDA_COMPONENTS_REDUCTION_CUH_


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/base/array_access.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {


constexpr int default_reduce_block_size = 512;


#include "common/cuda_hip/components/reduction.hpp.inc"


/**
 * Compute a reduction using add operation (+).
 *
 * @param exec  Executor associated to the array
 * @param size  size of the array
 * @param source  the pointer of the array
 *
 * @return the reduction result
 */
template <typename ValueType>
__host__ ValueType reduce_add_array(std::shared_ptr<const CudaExecutor> exec,
                                    size_type size, const ValueType* source)
{
    auto block_results_val = source;
    size_type grid_dim = size;
    auto block_results = array<ValueType>(exec);
    if (size > default_reduce_block_size) {
        const auto n = ceildiv(size, default_reduce_block_size);
        grid_dim =
            (n <= default_reduce_block_size) ? n : default_reduce_block_size;

        block_results.resize_and_reset(grid_dim);

        reduce_add_array<<<grid_dim, default_reduce_block_size, 0,
                           exec->get_stream()>>>(
            size, as_device_type(source),
            as_device_type(block_results.get_data()));

        block_results_val = block_results.get_const_data();
    }

    auto d_result = array<ValueType>(exec, 1);

    reduce_add_array<<<1, default_reduce_block_size, 0, exec->get_stream()>>>(
        grid_dim, as_device_type(block_results_val),
        as_device_type(d_result.get_data()));
    auto answer = get_element(d_result, 0);
    return answer;
}


}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_REDUCTION_CUH_
