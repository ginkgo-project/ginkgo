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

#include "core/matrix/diagonal_kernels.hpp"


#include <algorithm>


#include <omp.h>


#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Diagonal matrix format namespace.
 *
 * @ingroup diagonal
 */
namespace diagonal {


constexpr auto default_block_size = 512;


#include "common/matrix/diagonal_kernels.hpp.inc"


template <typename ValueType>
void apply_to_dense(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Diagonal<ValueType> *a,
                    const matrix::Dense<ValueType> *b,
                    matrix::Dense<ValueType> *c)
{
    const auto b_size = b->get_size();
    const auto num_rows = b_size[0];
    const auto num_cols = b_size[1];
    const auto b_stride = b->get_stride();
    const auto c_stride = c->get_stride();
    const auto grid_dim = ceildiv(num_rows * num_cols, default_block_size);

    const auto diag_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    auto c_values = c->get_values();

    hipLaunchKernelGGL(kernel::apply_to_dense, dim3(grid_dim),
                       dim3(default_block_size), 0, 0, num_rows, num_cols,
                       as_hip_type(diag_values), b_stride,
                       as_hip_type(b_values), c_stride, as_hip_type(c_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(GKO_DECLARE_DIAGONAL_APPLY_TO_DENSE_KERNEL);


template <typename ValueType>
void right_apply_to_dense(std::shared_ptr<const HipExecutor> exec,
                          const matrix::Diagonal<ValueType> *a,
                          const matrix::Dense<ValueType> *b,
                          matrix::Dense<ValueType> *c)
{
    const auto b_size = b->get_size();
    const auto num_rows = b_size[0];
    const auto num_cols = b_size[1];
    const auto b_stride = b->get_stride();
    const auto c_stride = c->get_stride();
    const auto grid_dim = ceildiv(num_rows * num_cols, default_block_size);

    const auto diag_values = a->get_const_values();
    const auto b_values = b->get_const_values();
    auto c_values = c->get_values();

    hipLaunchKernelGGL(kernel::right_apply_to_dense, dim3(grid_dim),
                       dim3(default_block_size), 0, 0, num_rows, num_cols,
                       as_hip_type(diag_values), b_stride,
                       as_hip_type(b_values), c_stride, as_hip_type(c_values));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void apply_to_csr(std::shared_ptr<const HipExecutor> exec,
                  const matrix::Diagonal<ValueType, IndexType> *a,
                  const matrix::Csr<ValueType, IndexType> *b,
                  matrix::Csr<ValueType, IndexType> *c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_APPLY_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void right_apply_to_csr(std::shared_ptr<const HipExecutor> exec,
                        const matrix::Diagonal<ValueType, IndexType> *a,
                        const matrix::Csr<ValueType, IndexType> *b,
                        matrix::Csr<ValueType, IndexType> *c)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_DIAGONAL_RIGHT_APPLY_TO_CSR_KERNEL);


}  // namespace diagonal
}  // namespace hip
}  // namespace kernels
}  // namespace gko
