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

#ifndef GKO_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_
#define GKO_HIP_COMPONENTS_FORMAT_CONVERSION_HPP_


#include <ginkgo/core/base/std_extensions.hpp>


namespace gko {
namespace kernels {
namespace hip {
namespace host_kernel {


/**
 * @internal
 *
 * It calculates the number of warps used in Coo Spmv by GPU architecture and
 * the number of stored elements.
 */
template <size_type subwarp_size = hip_config::warp_size>
__host__ size_type calculate_nwarps(std::shared_ptr<const HipExecutor> exec,
                                    const size_type nnz)
{
    // One multiprocessor has 4 SIMD
    size_type nwarps_in_hip = exec->get_num_multiprocessor() * 4;
    size_type multiple = 8;
    if (nnz >= 2000000) {
        multiple = 128;
    } else if (nnz >= 200000) {
        multiple = 32;
    }
    return std::min(multiple * nwarps_in_hip, static_cast<size_type>(ceildiv(
                                                  nnz, hip_config::warp_size)));
}


}  // namespace host_kernel
}  // namespace hip
}  // namespace kernels
}  // namespace gko


#endif  // GKO_CUDA_COMPONENTS_FORMAT_CONVERSION_CUH_
