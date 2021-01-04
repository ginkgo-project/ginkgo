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

#include "core/components/fill_array.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/math.hpp>


#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
namespace components {

constexpr int default_block_size = 512;


#include "common/components/fill_array.hpp.inc"


template <typename ValueType>
void fill_array(std::shared_ptr<const DefaultExecutor> exec, ValueType *array,
                size_type n, ValueType val)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
    hipLaunchKernelGGL(kernel::fill_array, dim3(grid_size), dim3(block_size), 0,
                       0, n, as_hip_type(array), as_hip_type(val));
}


GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_FILL_ARRAY_KERNEL);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_FILL_ARRAY_KERNEL);
template GKO_DECLARE_FILL_ARRAY_KERNEL(size_type);


}  // namespace components
}  // namespace hip
}  // namespace kernels
}  // namespace gko
