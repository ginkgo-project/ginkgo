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

#ifndef GKO_CUDA_COMPONENTS_FORMAT_CONVERSION_CUH_
#define GKO_CUDA_COMPONENTS_FORMAT_CONVERSION_CUH_


#include <ginkgo/core/base/std_extensions.hpp>


#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
namespace ell {
namespace kernel {


template <typename ValueType, typename IndexType>
__global__ void count_nnz_per_row(size_type num_rows, size_type max_nnz_per_row,
                                  size_type stride,
                                  const ValueType *__restrict__ values,
                                  IndexType *__restrict__ result);


}  // namespace kernel
}  // namespace ell


namespace coo {
namespace kernel {


template <typename IndexType>
__global__ void convert_row_idxs_to_ptrs(const IndexType *__restrict__ idxs,
                                         size_type num_nonzeros,
                                         IndexType *__restrict__ ptrs,
                                         size_type length);


}  // namespace kernel


namespace host_kernel {


template <size_type subwarp_size = cuda_config::warp_size>
size_type calculate_nwarps(std::shared_ptr<const CudaExecutor> exec,
                           const size_type nnz)
{
    size_type warps_per_sm = exec->get_num_cores_per_sm() / subwarp_size;
    size_type nwarps_in_cuda = exec->get_num_multiprocessor() * warps_per_sm;
    size_type multiple = 8;
    if (nnz >= 2000000) {
        multiple = 128;
    } else if (nnz >= 200000) {
        multiple = 32;
    }
    return std::min(
        multiple * nwarps_in_cuda,
        static_cast<size_type>(ceildiv(nnz, cuda_config::warp_size)));
}


}  // namespace host_kernel
}  // namespace coo
}  // namespace cuda
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_FORMAT_CONVERSION_CUH_
