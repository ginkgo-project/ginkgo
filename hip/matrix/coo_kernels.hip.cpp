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

#include "core/matrix/coo_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/format_conversion.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The HIP namespace.
 *
 * @ingroup hip
 */
namespace hip {
/**
 * @brief The Coordinate matrix format namespace.
 *
 * @ingroup coo
 */
namespace coo {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


#include "common/matrix/coo_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::Coo<ValueType, IndexType> *a,
          const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    components::fill_array(exec, c->get_values(), zero<ValueType>(),
                           c->get_num_stored_elements());

    spmv2(exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Dense<ValueType> *alpha,
                   const matrix::Coo<ValueType, IndexType> *a,
                   const matrix::Dense<ValueType> *b,
                   const matrix::Dense<ValueType> *beta,
                   matrix::Dense<ValueType> *c)
{
    dense::scale(exec, beta, c);
    advanced_spmv2(exec, alpha, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const HipExecutor> exec,
           const matrix::Coo<ValueType, IndexType> *a,
           const matrix::Dense<ValueType> *b, matrix::Dense<ValueType> *c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto b_ncols = b->get_size()[1];
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);

    if (nwarps > 0) {
        // TODO: b_ncols needs to be tuned.
        if (b_ncols < 4) {
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            hipLaunchKernelGGL(
                abstract_spmv, dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_lines, as_hip_type(a->get_const_values()),
                a->get_const_col_idxs(), as_hip_type(a->get_const_row_idxs()),
                as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));
            hipLaunchKernelGGL(
                abstract_spmm, dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_elems, as_hip_type(a->get_const_values()),
                a->get_const_col_idxs(), as_hip_type(a->get_const_row_idxs()),
                b_ncols, as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_COO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Dense<ValueType> *alpha,
                    const matrix::Coo<ValueType, IndexType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    const auto nnz = a->get_num_stored_elements();
    const auto nwarps = host_kernel::calculate_nwarps(exec, nnz);
    const dim3 coo_block(config::warp_size, warps_in_block, 1);
    const auto b_ncols = b->get_size()[1];

    if (nwarps > 0) {
        // TODO: b_ncols needs to be tuned.
        if (b_ncols < 4) {
            int num_lines = ceildiv(nnz, nwarps * config::warp_size);
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block), b_ncols);
            hipLaunchKernelGGL(
                abstract_spmv, dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_lines, as_hip_type(alpha->get_const_values()),
                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
                as_hip_type(a->get_const_row_idxs()),
                as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        } else {
            int num_elems =
                ceildiv(nnz, nwarps * config::warp_size) * config::warp_size;
            const dim3 coo_grid(ceildiv(nwarps, warps_in_block),
                                ceildiv(b_ncols, config::warp_size));
            hipLaunchKernelGGL(
                abstract_spmm, dim3(coo_grid), dim3(coo_block), 0, 0, nnz,
                num_elems, as_hip_type(alpha->get_const_values()),
                as_hip_type(a->get_const_values()), a->get_const_col_idxs(),
                as_hip_type(a->get_const_row_idxs()), b_ncols,
                as_hip_type(b->get_const_values()), b->get_stride(),
                as_hip_type(c->get_values()), c->get_stride());
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_ADVANCED_SPMV2_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const HipExecutor> exec,
                              const IndexType *idxs, size_type num_nonzeros,
                              IndexType *ptrs, size_type length)
{
    const auto grid_dim = ceildiv(num_nonzeros, default_block_size);

    hipLaunchKernelGGL(kernel::convert_row_idxs_to_ptrs, dim3(grid_dim),
                       dim3(default_block_size), 0, 0, as_hip_type(idxs),
                       num_nonzeros, as_hip_type(ptrs), length);
}


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Coo<ValueType, IndexType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
{
    auto num_rows = result->get_size()[0];

    auto row_ptrs = result->get_row_ptrs();
    const auto nnz = result->get_num_stored_elements();

    const auto source_row_idxs = source->get_const_row_idxs();

    convert_row_idxs_to_ptrs(exec, source_row_idxs, nnz, row_ptrs,
                             num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
                      const matrix::Coo<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result)
{
    const auto num_rows = result->get_size()[0];
    const auto num_cols = result->get_size()[1];
    const auto stride = result->get_stride();

    const auto nnz = source->get_num_stored_elements();

    const dim3 block_size(config::warp_size,
                          config::max_block_size / config::warp_size, 1);
    const dim3 init_grid_dim(ceildiv(stride, block_size.x),
                             ceildiv(num_rows, block_size.y), 1);
    hipLaunchKernelGGL(kernel::initialize_zero_dense, dim3(init_grid_dim),
                       dim3(block_size), 0, 0, num_rows, num_cols, stride,
                       as_hip_type(result->get_values()));

    const auto grid_dim = ceildiv(nnz, default_block_size);
    hipLaunchKernelGGL(kernel::fill_in_dense, dim3(grid_dim),
                       dim3(default_block_size), 0, 0, nnz,
                       as_hip_type(source->get_const_row_idxs()),
                       as_hip_type(source->get_const_col_idxs()),
                       as_hip_type(source->get_const_values()), stride,
                       as_hip_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_COO_CONVERT_TO_DENSE_KERNEL);


}  // namespace coo
}  // namespace hip
}  // namespace kernels
}  // namespace gko
