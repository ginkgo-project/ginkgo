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

#include "core/factorization/par_ilu_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/matrix/coo.hpp>


#include "core/components/prefix_sum.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ilu factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilu_factorization {


constexpr int default_block_size{512};


#include "common/factorization/par_ilu_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void add_diagonal_elements(std::shared_ptr<const DefaultExecutor> exec,
                           matrix::Csr<ValueType, IndexType> *mtx,
                           bool is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_ADD_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *system_matrix,
    IndexType *l_row_ptrs, IndexType *u_row_ptrs)
{
    const size_type num_rows{system_matrix->get_size()[0]};

    const dim3 block_size{default_block_size, 1, 1};
    const uint32 number_blocks =
        ceildiv(num_rows, static_cast<size_type>(block_size.x));
    const dim3 grid_dim{number_blocks, 1, 1};

    hipLaunchKernelGGL(kernel::count_nnz_per_l_u_row, dim3(grid_dim),
                       dim3(block_size), 0, 0, num_rows,
                       as_hip_type(system_matrix->get_const_row_ptrs()),
                       as_hip_type(system_matrix->get_const_col_idxs()),
                       as_hip_type(system_matrix->get_const_values()),
                       as_hip_type(l_row_ptrs), as_hip_type(u_row_ptrs));

    prefix_sum(exec, l_row_ptrs, num_rows + 1);
    prefix_sum(exec, u_row_ptrs, num_rows + 1);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_INITIALIZE_ROW_PTRS_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_l_u(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *system_matrix,
                    matrix::Csr<ValueType, IndexType> *csr_l,
                    matrix::Csr<ValueType, IndexType> *csr_u)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const dim3 block_size{default_block_size, 1, 1};
    const dim3 grid_dim{static_cast<uint32>(ceildiv(
                            num_rows, static_cast<size_type>(block_size.x))),
                        1, 1};

    hipLaunchKernelGGL(
        kernel::initialize_l_u, dim3(grid_dim), dim3(block_size), 0, 0,
        num_rows, as_hip_type(system_matrix->get_const_row_ptrs()),
        as_hip_type(system_matrix->get_const_col_idxs()),
        as_hip_type(system_matrix->get_const_values()),
        as_hip_type(csr_l->get_const_row_ptrs()),
        as_hip_type(csr_l->get_col_idxs()), as_hip_type(csr_l->get_values()),
        as_hip_type(csr_u->get_const_row_ptrs()),
        as_hip_type(csr_u->get_col_idxs()), as_hip_type(csr_u->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_INITIALIZE_L_U_KERNEL);


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const HipExecutor> exec,
                         size_type iterations,
                         const matrix::Coo<ValueType, IndexType> *system_matrix,
                         matrix::Csr<ValueType, IndexType> *l_factor,
                         matrix::Csr<ValueType, IndexType> *u_factor)
{
    iterations = (iterations == 0) ? 10 : iterations;
    const auto num_elements = system_matrix->get_num_stored_elements();
    const dim3 block_size{default_block_size, 1, 1};
    const dim3 grid_dim{
        static_cast<uint32>(
            ceildiv(num_elements, static_cast<size_type>(block_size.x))),
        1, 1};
    for (size_type i = 0; i < iterations; ++i) {
        hipLaunchKernelGGL(kernel::compute_l_u_factors, dim3(grid_dim),
                           dim3(block_size), 0, 0, num_elements,
                           as_hip_type(system_matrix->get_const_row_idxs()),
                           as_hip_type(system_matrix->get_const_col_idxs()),
                           as_hip_type(system_matrix->get_const_values()),
                           as_hip_type(l_factor->get_const_row_ptrs()),
                           as_hip_type(l_factor->get_const_col_idxs()),
                           as_hip_type(l_factor->get_values()),
                           as_hip_type(u_factor->get_const_row_ptrs()),
                           as_hip_type(u_factor->get_const_col_idxs()),
                           as_hip_type(u_factor->get_values()));
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_COMPUTE_L_U_FACTORS_KERNEL);


}  // namespace par_ilu_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
