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

#include "core/matrix/coo_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/cusparse_bindings.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
/**
 * @brief The CUDA namespace.
 *
 * @ingroup cuda
 */
namespace cuda {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


#include "common/cuda_hip/matrix/coo_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const CudaExecutor> exec,
          const matrix::Coo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    dense::fill(exec, c, zero<ValueType>());
    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const CudaExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Coo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const CudaExecutor> exec,
           const matrix::Coo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto b_ncols = b->get_size()[1];
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0 && b_ncols > 0) {
        if (b_ncols < 4) {
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            abstract_spmv<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_lines, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));
            abstract_spmm<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_elems, as_cuda_type(a->get_const_values()),
                a->get_const_col_idxs(), as_cuda_type(a->get_const_row_idxs()),
                b_ncols, as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Coo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto b_ncols = b->get_size()[1];

    if (nwarps > 0 && b_ncols > 0) {
        if (b_ncols < 4) {
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            abstract_spmv<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_lines, as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_idxs()),
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));
            abstract_spmm<<<coo_grid, coo_block, 0, exec->get_stream()>>>(
                nnz, num_elems, as_cuda_type(alpha->get_const_values()),
                as_cuda_type(a->get_const_values()), a->get_const_col_idxs(),
                as_cuda_type(a->get_const_row_idxs()), b_ncols,
                as_cuda_type(b->get_const_values()), b->get_stride(),
                as_cuda_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


}  // namespace coo
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
