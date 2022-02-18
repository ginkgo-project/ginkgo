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

#include "core/factorization/par_ic_kernels.hpp"


#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel IC factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ic_factorization {


constexpr int default_block_size = 512;


// subwarp sizes for all warp-parallel kernels (sweep)
using compiled_kernels =
    std::integer_sequence<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/cuda_hip/factorization/par_ic_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void init_factor(std::shared_ptr<const DefaultExecutor> exec,
                 matrix::Csr<ValueType, IndexType>* l)
{
    auto num_rows = l->get_size()[0];
    auto num_blocks = ceildiv(num_rows, default_block_size);
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_vals = l->get_values();
    if (num_blocks > 0) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::ic_init), num_blocks,
                           default_block_size, 0, 0, l_row_ptrs,
                           as_hip_type(l_vals), num_rows);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_INIT_FACTOR_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    size_type iterations,
                    const matrix::Coo<ValueType, IndexType>* a_lower,
                    matrix::Csr<ValueType, IndexType>* l)
{
    auto nnz = l->get_num_stored_elements();
    auto num_blocks = ceildiv(nnz, default_block_size);
    if (num_blocks > 0) {
        for (size_type i = 0; i < iterations; ++i) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(kernel::ic_sweep), num_blocks,
                default_block_size, 0, 0, a_lower->get_const_row_idxs(),
                a_lower->get_const_col_idxs(),
                as_hip_type(a_lower->get_const_values()),
                l->get_const_row_ptrs(), l->get_const_col_idxs(),
                as_hip_type(l->get_values()),
                static_cast<IndexType>(l->get_num_stored_elements()));
        }
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_IC_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ic_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
