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

#include "core/matrix/hybrid_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/format_conversion.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/zero_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Hybrid matrix format namespace.
 *
 * @ingroup hybrid
 */
namespace hybrid {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;


#include "common/matrix/hybrid_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const HipExecutor> exec, matrix::Dense<ValueType> *result,
    const matrix::Hybrid<ValueType, IndexType> *source) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const HipExecutor> exec,
                    matrix::Csr<ValueType, IndexType> *result,
                    const matrix::Hybrid<ValueType, IndexType> *source)
{
    const auto num_rows = source->get_size()[0];
    auto coo_offset = Array<IndexType>(exec, num_rows + 1);
    auto coo_val = source->get_const_coo_values();
    auto coo_col = source->get_const_coo_col_idxs();
    auto coo_row = source->get_const_coo_row_idxs();
    auto ell_val = source->get_const_ell_values();
    auto ell_col = source->get_const_ell_col_idxs();
    const auto stride = source->get_ell_stride();
    const auto max_nnz_per_row = source->get_ell_num_stored_elements_per_row();
    const auto coo_num_stored_elements = source->get_coo_num_stored_elements();

    // Compute the row offset of Coo without zeros
    size_type grid_num = ceildiv(coo_num_stored_elements, default_block_size);
    hipLaunchKernelGGL(coo::kernel::convert_row_idxs_to_ptrs, dim3(grid_num),
                       dim3(default_block_size), 0, 0, as_hip_type(coo_row),
                       coo_num_stored_elements,
                       as_hip_type(coo_offset.get_data()), num_rows + 1);

    // Compute the row ptrs of Csr
    auto row_ptrs = result->get_row_ptrs();
    auto coo_row_ptrs = Array<IndexType>(exec, num_rows);

    zero_array(num_rows + 1, row_ptrs);
    grid_num = ceildiv(num_rows, warps_in_block);
    hipLaunchKernelGGL(ell::kernel::count_nnz_per_row, dim3(grid_num),
                       dim3(default_block_size), 0, 0, num_rows,
                       max_nnz_per_row, stride, as_hip_type(ell_val),
                       as_hip_type(row_ptrs));

    zero_array(num_rows, coo_row_ptrs.get_data());

    auto nwarps =
        coo::host_kernel::calculate_nwarps(exec, coo_num_stored_elements);
    if (nwarps > 0) {
        int num_lines =
            ceildiv(coo_num_stored_elements, nwarps * config::warp_size);
        const dim3 coo_block(config::warp_size, warps_in_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_in_block), 1);

        hipLaunchKernelGGL(
            kernel::count_coo_row_nnz, dim3(coo_grid), dim3(coo_block), 0, 0,
            coo_num_stored_elements, num_lines, as_hip_type(coo_val),
            as_hip_type(coo_row), as_hip_type(coo_row_ptrs.get_data()));
    }

    hipLaunchKernelGGL(kernel::add, dim3(grid_num), dim3(default_block_size), 0,
                       0, num_rows, as_hip_type(row_ptrs),
                       as_hip_type(coo_row_ptrs.get_const_data()));

    prefix_sum(exec, row_ptrs, num_rows + 1);

    // Fill the value
    grid_num = ceildiv(num_rows, default_block_size);
    hipLaunchKernelGGL(
        kernel::fill_in_csr, dim3(grid_num), dim3(default_block_size), 0, 0,
        num_rows, max_nnz_per_row, stride, as_hip_type(ell_val),
        as_hip_type(ell_col), as_hip_type(coo_val), as_hip_type(coo_col),
        as_hip_type(coo_offset.get_const_data()), as_hip_type(row_ptrs),
        as_hip_type(result->get_col_idxs()), as_hip_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Hybrid<ValueType, IndexType> *source,
                    size_type *result)
{
    size_type ell_nnz = 0;
    size_type coo_nnz = 0;
    ell::count_nonzeros(exec, source->get_ell(), &ell_nnz);

    auto nnz = source->get_coo_num_stored_elements();
    auto nwarps = coo::host_kernel::calculate_nwarps(exec, nnz);
    if (nwarps > 0) {
        int num_lines = ceildiv(nnz, nwarps * config::warp_size);
        const dim3 coo_block(config::warp_size, warps_in_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_in_block), 1);
        const auto num_rows = source->get_size()[0];
        auto nnz_per_row = Array<IndexType>(exec, num_rows);
        zero_array(num_rows, nnz_per_row.get_data());
        hipLaunchKernelGGL(kernel::count_coo_row_nnz, dim3(coo_grid),
                           dim3(coo_block), 0, 0, nnz, num_lines,
                           as_hip_type(source->get_coo()->get_const_values()),
                           as_hip_type(source->get_coo()->get_const_row_idxs()),
                           as_hip_type(nnz_per_row.get_data()));

        coo_nnz =
            reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
    }

    *result = ell_nnz + coo_nnz;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_COUNT_NONZEROS_KERNEL);


}  // namespace hybrid
}  // namespace hip
}  // namespace kernels
}  // namespace gko
