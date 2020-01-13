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

#include "core/components/prefix_sum.hpp"


#include "hip/components/prefix_sum.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {


constexpr int prefix_sum_block_size = 512;


template <typename IndexType>
void prefix_sum(std::shared_ptr<const HipExecutor> exec, IndexType *counts,
                size_type num_entries)
{
    auto num_blocks = ceildiv(num_entries, prefix_sum_block_size);
    Array<IndexType> block_sum_array(exec, num_blocks);
    auto block_sums = block_sum_array.get_data();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(start_prefix_sum<prefix_sum_block_size>),
                       dim3(num_blocks), dim3(prefix_sum_block_size), 0, 0,
                       num_entries, counts, block_sums);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(finalize_prefix_sum<prefix_sum_block_size>),
        dim3(num_blocks), dim3(prefix_sum_block_size), 0, 0, num_entries,
        counts, block_sums);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PREFIX_SUM_KERNEL);

// explicitly instantiate for size_type as well, as this is used in the SellP
// format
template void prefix_sum<size_type>(std::shared_ptr<const HipExecutor> exec,
                                    size_type *counts, size_type num_entries);


}  // namespace hip
}  // namespace kernels
}  // namespace gko
