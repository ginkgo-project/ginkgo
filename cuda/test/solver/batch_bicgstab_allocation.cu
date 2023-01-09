/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#include <gtest/gtest.h>


#include "core/matrix/batch_struct.hpp"
#include "core/solver/batch_bicgstab_kernels.hpp"
#include "cuda/base/config.hpp"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/matrix/batch_struct.hpp"


namespace gko {
namespace kernels {
namespace cuda {
#include "common/cuda_hip/preconditioner/batch_jacobi.hpp.inc"
}
}  // namespace kernels
}  // namespace gko


TEST(BatchBicgstab, CanAssignVectorsToGlobalMemory)
{
    using T = double;
    using PC = gko::kernels::cuda::BatchJacobi<T>;
    const int nrows = 5;
    const int nrhs = 1;
    const int nnz = 16;
    size_t shmem_per_sm = 4;
    const int batch_storage = 10 * nrows * sizeof(T);

    const auto conf =
        gko::kernels::batch_bicgstab::compute_shared_storage<PC, T>(
            shmem_per_sm, nrows, nnz, nrhs);

    ASSERT_FALSE(conf.prec_shared);
    ASSERT_EQ(conf.n_shared, 0);
    ASSERT_EQ(conf.n_global, 9);
    ASSERT_EQ(conf.gmem_stride_bytes, ((batch_storage - 1) / 32 + 1) * 32);
}

TEST(BatchBicgstab, AssignsPriorityVectorsToSharedMemoryFirst)
{
    using T = double;
    using PC = gko::kernels::cuda::BatchJacobi<T>;
    const int nrows = 5;
    const int nrhs = 1;
    const int nnz = 16;
    int shmem_per_sm = (2 * nrows) * sizeof(T);
    const int gmem_batch_storage = 8 * nrows * sizeof(T);

    const auto conf =
        gko::kernels::batch_bicgstab::compute_shared_storage<PC, T>(
            shmem_per_sm, nrows, nnz, nrhs);

    ASSERT_FALSE(conf.prec_shared);
    ASSERT_EQ(conf.n_shared, 2);
    ASSERT_EQ(conf.n_global, 7);
    ASSERT_EQ(conf.gmem_stride_bytes, ((gmem_batch_storage - 1) / 32 + 1) * 32);
}

TEST(BatchBicgstab, CanAssignAllVectorsToSharedMemory)
{
    using T = double;
    using PC = gko::kernels::cuda::BatchJacobi<T>;
    const int nrows = 5;
    const int nrhs = 1;
    const int nnz = 16;
    const int shmem_per_sm = (10 * nrows) * sizeof(T);

    const auto conf =
        gko::kernels::batch_bicgstab::compute_shared_storage<PC, T>(
            shmem_per_sm, nrows, nnz, nrhs);

    ASSERT_EQ(conf.n_shared, 9);
    ASSERT_EQ(conf.n_global, 0);
    ASSERT_EQ(conf.gmem_stride_bytes, 0);
    ASSERT_TRUE(conf.prec_shared);
}

TEST(BatchBicgstab, AssignsMultipleRHSCorrectly)
{
    using T = double;
    using PC = gko::kernels::cuda::BatchJacobi<T>;
    const int nrows = 5;
    const int nrhs = 3;
    const int nnz = 16;
    int shmem_per_sm = 3 * nrhs * nrows * sizeof(T);

    const auto conf =
        gko::kernels::batch_bicgstab::compute_shared_storage<PC, T>(
            shmem_per_sm, nrows, nnz, nrhs);

    ASSERT_EQ(conf.n_shared, 3);
    ASSERT_EQ(conf.n_global, 6);
    ASSERT_FALSE(conf.prec_shared);
    ASSERT_EQ(conf.gmem_stride_bytes,
              (((6 * nrows * nrhs + nrows) * sizeof(T) - 1) / 32 + 1) * 32);
}
