/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2019, the Ginkgo authors
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

#ifndef GKO_HIP_COMPONENTS_REDUCTION_HIP_HPP_
#define GKO_HIP_COMPONENTS_REDUCTION_HIP_HPP_


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/std_extensions.hpp>


#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


constexpr int default_block_size = 512;


#include "common/components/reduction.hpp.inc"


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
__host__ ValueType reduce_add_array(std::shared_ptr<const HipExecutor> exec,
                                    size_type size, const ValueType *source)
{
    auto block_results_val = source;
    size_type grid_dim = size;
    if (size > default_block_size) {
        const auto n = ceildiv(size, default_block_size);
        grid_dim = (n <= default_block_size) ? n : default_block_size;

        auto block_results = Array<ValueType>(exec, grid_dim);

        hipLaunchKernelGGL(
            reduce_add_array, dim3(grid_dim), dim3(default_block_size), 0, 0,
            size, as_hip_type(source), as_hip_type(block_results.get_data()));

        block_results_val = block_results.get_const_data();
    }

    auto d_result = Array<ValueType>(exec, 1);

    hipLaunchKernelGGL(reduce_add_array, dim3(1), dim3(default_block_size), 0,
                       0, grid_dim, as_hip_type(block_results_val),
                       as_hip_type(d_result.get_data()));
    ValueType answer = zero<ValueType>();
    exec->get_master()->get_mem_space()->copy_from(
        exec->get_mem_space().get(), 1, d_result.get_const_data(), &answer);
    return answer;
}


}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_HIP_COMPONENTS_REDUCTION_HIP_HPP_
