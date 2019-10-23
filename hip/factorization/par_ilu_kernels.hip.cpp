#include "hip/hip_runtime.h"
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

#include "core/factorization/par_ilu_kernels.hpp"


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/std_extensions.hpp>
#include <ginkgo/core/matrix/coo.hpp>


#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"


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


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void count_nnz_per_l_u_row(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values, IndexType *__restrict__ l_nnz_row,
    IndexType *__restrict__ u_nnz_row)
{
    const auto row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        IndexType l_row_nnz{};
        IndexType u_row_nnz{};
        for (auto idx = row_ptrs[row]; idx < row_ptrs[row + 1]; ++idx) {
            auto col = col_idxs[idx];
            l_row_nnz += (col <= row);
            u_row_nnz += (row <= col);
        }
        l_nnz_row[row] = l_row_nnz;
        u_nnz_row[row] = u_row_nnz;
    }
}


}  // namespace kernel


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_l_u(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Csr<ValueType, IndexType> *system_matrix,
    IndexType *l_row_ptrs, IndexType *u_row_ptrs)
{
    const size_type num_rows{system_matrix->get_size()[0]};
    const size_type num_row_ptrs{num_rows + 1};

    const dim3 block_size{default_block_size, 1, 1};
    const uint32 number_blocks =
        ceildiv(num_rows, static_cast<size_type>(block_size.x));
    const dim3 grid_dim{number_blocks, 1, 1};

    hipLaunchKernelGGL(kernel::count_nnz_per_l_u_row, dim3(grid_dim), dim3(block_size), 0, 0, 
        num_rows, as_hip_type(system_matrix->get_const_row_ptrs()),
        as_hip_type(system_matrix->get_const_col_idxs()),
        as_hip_type(system_matrix->get_const_values()),
        as_hip_type(l_row_ptrs), as_hip_type(u_row_ptrs));

    Array<IndexType> block_sum(exec, grid_dim.x);
    auto block_sum_ptr = block_sum.get_data();

    hipLaunchKernelGGL(HIP_KERNEL_NAME(start_prefix_sum<default_block_size>), dim3(grid_dim), dim3(block_size), 0, 0, 
        num_row_ptrs, as_hip_type(l_row_ptrs), as_hip_type(block_sum_ptr));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(finalize_prefix_sum<default_block_size>), dim3(grid_dim), dim3(block_size), 0, 0, 
        num_row_ptrs, as_hip_type(l_row_ptrs), as_hip_type(block_sum_ptr));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(start_prefix_sum<default_block_size>), dim3(grid_dim), dim3(block_size), 0, 0, 
        num_row_ptrs, as_hip_type(u_row_ptrs), as_hip_type(block_sum_ptr));
    hipLaunchKernelGGL(HIP_KERNEL_NAME(finalize_prefix_sum<default_block_size>), dim3(grid_dim), dim3(block_size), 0, 0, 
        num_row_ptrs, as_hip_type(u_row_ptrs), as_hip_type(block_sum_ptr));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILU_INITIALIZE_ROW_PTRS_L_U_KERNEL);


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void initialize_l_u(
    size_type num_rows, const IndexType *__restrict__ row_ptrs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values,
    const IndexType *__restrict__ l_row_ptrs,
    IndexType *__restrict__ l_col_idxs, ValueType *__restrict__ l_values,
    const IndexType *__restrict__ u_row_ptrs,
    IndexType *__restrict__ u_col_idxs, ValueType *__restrict__ u_values)
{
    const auto row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[row];
        for (size_type i = row_ptrs[row]; i < row_ptrs[row + 1]; ++i) {
            const auto col = col_idxs[i];
            const auto val = values[i];
            if (col <= row) {
                l_col_idxs[l_idx] = col;
                l_values[l_idx] = (col == row ? one<ValueType>() : val);
                ++l_idx;
            }
            if (row <= col) {
                u_col_idxs[u_idx] = col;
                u_values[u_idx] = val;
                ++u_idx;
            }
        }
    }
}


}  // namespace kernel


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

    hipLaunchKernelGGL(kernel::initialize_l_u, dim3(grid_dim), dim3(block_size), 0, 0, 
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


namespace kernel {


template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(default_block_size) void compute_l_u_factors(
    size_type num_elements, const IndexType *__restrict__ row_idxs,
    const IndexType *__restrict__ col_idxs,
    const ValueType *__restrict__ values,
    const IndexType *__restrict__ l_row_ptrs,
    const IndexType *__restrict__ l_col_idxs, ValueType *__restrict__ l_values,
    const IndexType *__restrict__ u_row_ptrs,
    const IndexType *__restrict__ u_col_idxs, ValueType *__restrict__ u_values)
{
    const auto elem_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (elem_id < num_elements) {
        const auto row = row_idxs[elem_id];
        const auto col = col_idxs[elem_id];
        const auto val = values[elem_id];
        auto l_idx = l_row_ptrs[row];
        auto u_idx = u_row_ptrs[col];
        ValueType sum{val};
        ValueType last_operation{};
        while (l_idx < l_row_ptrs[row + 1] && u_idx < u_row_ptrs[col + 1]) {
            const auto l_col = l_col_idxs[l_idx];
            const auto u_col = u_col_idxs[u_idx];
            last_operation = zero<ValueType>();
            if (l_col == u_col) {
                last_operation = l_values[l_idx] * u_values[u_idx];
                sum -= last_operation;
            }
            l_idx += (l_col <= u_col);
            u_idx += (u_col <= l_col);
        }
        sum += last_operation;  // undo the last operation
        if (row > col) {
            auto to_write = sum / u_values[u_row_ptrs[col + 1] - 1];
            if (::gko::isfinite(to_write)) {
                l_values[l_idx - 1] = to_write;
            }
        } else {
            auto to_write = sum;
            if (::gko::isfinite(to_write)) {
                u_values[u_idx - 1] = to_write;
            }
        }
    }
}


}  // namespace kernel


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
        hipLaunchKernelGGL(kernel::compute_l_u_factors, dim3(grid_dim), dim3(block_size), 0, 0, 
            num_elements, as_hip_type(system_matrix->get_const_row_idxs()),
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
