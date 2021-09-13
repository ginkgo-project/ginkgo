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

#include "core/components/reduce_array.hpp"


#include <type_traits>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/components/uninitialized_array.hpp"


namespace gko {
namespace kernels {
namespace cuda {
namespace components {


constexpr int default_block_size = 512;


#include "common/cuda_hip/components/reduction.hpp.inc"


template <typename ValueType>
void reduce_array(std::shared_ptr<const DefaultExecutor> exec,
                  const ValueType* array, size_type size, ValueType* val)
{
    auto block_results_val = array;
    size_type grid_dim = size;
    auto block_results = Array<ValueType>(exec);
    if (size > default_block_size) {
        const auto n = ceildiv(size, default_block_size);
        grid_dim = (n <= default_block_size) ? n : default_block_size;

        block_results.resize_and_reset(grid_dim);

        reduce_add_array<<<grid_dim, default_block_size>>>(
            size, as_cuda_type(array), as_cuda_type(block_results.get_data()));

        block_results_val = block_results.get_const_data();
    }

    auto d_result = Array<ValueType>::view(exec, 1, val);

    reduce_add_array_with_existing_value<<<1, default_block_size>>>(
        grid_dim, as_cuda_type(block_results_val),
        as_cuda_type(d_result.get_data()));
}

GKO_INSTANTIATE_FOR_EACH_TEMPLATE_TYPE(GKO_DECLARE_REDUCE_ARRAY_KERNEL);


}  // namespace components
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
