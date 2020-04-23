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

#include "core/preconditioner/isai_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/matrix/csr_builder.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Isai preconditioner namespace.
 * @ref Isai
 * @ingroup isai
 */
namespace isai {
namespace {


#include "common/preconditioner/isai_kernels.hpp.inc"


}  // namespace


constexpr int subwarp_size{config::warp_size};
constexpr int subwarps_per_block{1};
constexpr int default_block_size{subwarps_per_block * subwarp_size};


template <typename ValueType, typename IndexType>
void generate_l_inverse(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType> *l_csr,
                        matrix::Csr<ValueType, IndexType> *inverse_l)
{
    const auto nnz = l_csr->get_num_stored_elements();
    const auto num_rows = l_csr->get_size()[0];

    exec->copy_from(exec.get(), nnz, l_csr->get_const_col_idxs(),
                    inverse_l->get_col_idxs());
    exec->copy_from(exec.get(), num_rows + 1, l_csr->get_const_row_ptrs(),
                    inverse_l->get_row_ptrs());

    const dim3 block(default_block_size, 1, 1);
    const dim3 grid(ceildiv(num_rows, block.x / config::warp_size), 1, 1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            kernel::generate_l_inverse<subwarp_size, subwarps_per_block>),
        grid, block, 0, 0, static_cast<IndexType>(num_rows),
        l_csr->get_const_row_ptrs(), l_csr->get_const_col_idxs(),
        as_hip_type(l_csr->get_const_values()), inverse_l->get_row_ptrs(),
        inverse_l->get_col_idxs(), as_hip_type(inverse_l->get_values()));
    // Call make_srow()
    matrix::CsrBuilder<ValueType, IndexType> builder(inverse_l);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_L_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void generate_u_inverse(std::shared_ptr<const DefaultExecutor> exec,
                        const matrix::Csr<ValueType, IndexType> *u_csr,
                        matrix::Csr<ValueType, IndexType> *inverse_u)
{
    const auto nnz = u_csr->get_num_stored_elements();
    const auto num_rows = u_csr->get_size()[0];

    exec->copy_from(exec.get(), nnz, u_csr->get_const_col_idxs(),
                    inverse_u->get_col_idxs());
    exec->copy_from(exec.get(), num_rows + 1, u_csr->get_const_row_ptrs(),
                    inverse_u->get_row_ptrs());

    const dim3 block(default_block_size, 1, 1);
    const dim3 grid(ceildiv(num_rows, block.x / config::warp_size), 1, 1);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(
            kernel::generate_u_inverse<subwarp_size, subwarps_per_block>),
        grid, block, 0, 0, static_cast<IndexType>(num_rows),
        u_csr->get_const_row_ptrs(), u_csr->get_const_col_idxs(),
        as_hip_type(u_csr->get_const_values()), inverse_u->get_row_ptrs(),
        inverse_u->get_col_idxs(), as_hip_type(inverse_u->get_values()));
    // Call make_srow()
    matrix::CsrBuilder<ValueType, IndexType> builder(inverse_u);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_GENERATE_U_INVERSE_KERNEL);


template <typename ValueType, typename IndexType>
void identity_triangle(std::shared_ptr<const DefaultExecutor> exec,
                       matrix::Csr<ValueType, IndexType> *mtx,
                       bool lower) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ISAI_IDENTITY_TRIANGLE_KERNEL);


}  // namespace isai
}  // namespace hip
}  // namespace kernels
}  // namespace gko
