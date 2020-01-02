/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2020, the Ginkgo authors
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

#include "hip/components/zero_array.hip.hpp"


#include <hip/hip_runtime.h>


namespace gko {
namespace kernels {
namespace hip {


constexpr int default_block_size = 512;


#include "common/components/zero_array.hpp.inc"


template <typename ValueType>
void zero_array(size_type n, ValueType *array)
{
    const dim3 block_size(default_block_size, 1, 1);
    const dim3 grid_size(ceildiv(n, block_size.x), 1, 1);
    hipLaunchKernelGGL(kernel::zero_array, dim3(grid_size), dim3(block_size), 0,
                       0, n, as_hip_type(array));
}


#define GKO_DECLARE_ZERO_ARRAY(_type) \
    void zero_array<_type>(size_type n, _type * array);
GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_ZERO_ARRAY);
GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_ZERO_ARRAY);


}  // namespace hip
}  // namespace kernels
}  // namespace gko
