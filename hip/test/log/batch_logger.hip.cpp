/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include <hip/hip_runtime.h>


#include <gtest/gtest.h>


#include <ginkgo/core/base/exception.hpp>
#include <ginkgo/core/base/executor.hpp>


#include "core/matrix/batch_struct.hpp"
#include "core/test/utils.hpp"
#include "core/test/utils/batch.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {

constexpr int default_block_size = 128;

#include "common/cuda_hip/log/batch_logger.hpp.inc"

}  // namespace hip
}  // namespace kernels
}  // namespace gko


namespace {


template <typename RealType>
__global__ void simple_ex_iter(
    const size_t nbatch,
    gko::kernels::hip::batch_log::SimpleFinalLogger<RealType> blog,
    const int iter)
{
    for (size_t ib = blockIdx.x; ib < nbatch; ib += gridDim.x) {
        __shared__ RealType resnv;
        if (threadIdx.x == 0) {
            resnv = 1.2 + static_cast<RealType>(ib);
        }
        __syncthreads();
        blog.log_iteration(ib, iter + static_cast<int>(ib), resnv);
    }
}


TEST(BatchSimpleFinalLogger, Logs)
{
    using real_type = float;
    using BatchLog = gko::kernels::hip::batch_log::SimpleFinalLogger<real_type>;
    auto exec = gko::ReferenceExecutor::create();
    auto cuexec = gko::HipExecutor::create(0, exec);
    const size_t nbatch = 3;
    const int dbs = gko::kernels::hip::default_block_size;
    gko::array<real_type> res_norms_log(exec, nbatch);
    gko::array<int> iters_log(exec, nbatch);
    for (int i = 0; i < nbatch; i++) {
        res_norms_log.get_data()[i] = 0.0;
        iters_log.get_data()[i] = -1;
    }
    gko::array<real_type> d_res_norms_log(cuexec, res_norms_log);
    gko::array<int> d_iters_log(cuexec, iters_log);
    const int iter = 5;

    BatchLog blog(d_res_norms_log.get_data(), d_iters_log.get_data());
    hipLaunchKernelGGL(simple_ex_iter, dim3(nbatch), dim3(dbs), 0, 0, nbatch,
                       blog, iter);

    res_norms_log = d_res_norms_log;
    iters_log = d_iters_log;
    for (size_t i = 0; i < nbatch; i++) {
        ASSERT_LE((res_norms_log.get_const_data()[i] - 1.2 -
                   static_cast<real_type>(i)) /
                      (1.2 + static_cast<real_type>(i)),
                  r<real_type>::value);
        ASSERT_EQ(iters_log.get_const_data()[i], iter + static_cast<int>(i));
    }
}


}  // namespace
