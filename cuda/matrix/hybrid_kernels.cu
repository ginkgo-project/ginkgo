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


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/matrix/ell.hpp>


#include "core/components/fill_array.hpp"
#include "core/components/prefix_sum.hpp"
#include "core/matrix/coo_kernels.hpp"
#include "core/matrix/ell_kernels.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/format_conversion.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/segment_scan.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
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
void convert_to_dense(std::shared_ptr<const CudaExecutor> exec,
                      const matrix::Hybrid<ValueType, IndexType> *source,
                      matrix::Dense<ValueType> *result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const CudaExecutor> exec,
                    const matrix::Hybrid<ValueType, IndexType> *source,
                    matrix::Csr<ValueType, IndexType> *result)
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
    coo::kernel::convert_row_idxs_to_ptrs<<<grid_num, default_block_size>>>(
        as_cuda_type(coo_row), coo_num_stored_elements,
        as_cuda_type(coo_offset.get_data()), num_rows + 1);

    // Compute the row ptrs of Csr
    auto row_ptrs = result->get_row_ptrs();
    auto coo_row_ptrs = Array<IndexType>(exec, num_rows);

    components::fill_array(exec, row_ptrs, num_rows + 1, zero<IndexType>());
    grid_num = ceildiv(num_rows, warps_in_block);
    ell::kernel::count_nnz_per_row<<<grid_num, default_block_size>>>(
        num_rows, max_nnz_per_row, stride, as_cuda_type(ell_val),
        as_cuda_type(row_ptrs));

    components::fill_array(exec, coo_row_ptrs.get_data(), num_rows,
                           zero<IndexType>());

    auto nwarps =
        coo::host_kernel::calculate_nwarps(exec, coo_num_stored_elements);
    if (nwarps > 0) {
        int num_lines =
            ceildiv(coo_num_stored_elements, nwarps * config::warp_size);
        const dim3 coo_block(config::warp_size, warps_in_block, 1);
        const dim3 coo_grid(ceildiv(nwarps, warps_in_block), 1);

        kernel::count_coo_row_nnz<<<coo_grid, coo_block>>>(
            coo_num_stored_elements, num_lines, as_cuda_type(coo_val),
            as_cuda_type(coo_row), as_cuda_type(coo_row_ptrs.get_data()));
    }

    kernel::add<<<grid_num, default_block_size>>>(
        num_rows, as_cuda_type(row_ptrs),
        as_cuda_type(coo_row_ptrs.get_const_data()));

    components::prefix_sum(exec, row_ptrs, num_rows + 1);

    // Fill the value
    grid_num = ceildiv(num_rows, default_block_size);
    kernel::fill_in_csr<<<grid_num, default_block_size>>>(
        num_rows, max_nnz_per_row, stride, as_cuda_type(ell_val),
        as_cuda_type(ell_col), as_cuda_type(coo_val), as_cuda_type(coo_col),
        as_cuda_type(coo_offset.get_const_data()), as_cuda_type(row_ptrs),
        as_cuda_type(result->get_col_idxs()),
        as_cuda_type(result->get_values()));
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void count_nonzeros(std::shared_ptr<const CudaExecutor> exec,
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
        components::fill_array(exec, nnz_per_row.get_data(), num_rows,
                               zero<IndexType>());
        kernel::count_coo_row_nnz<<<coo_grid, coo_block>>>(
            nnz, num_lines, as_cuda_type(source->get_coo()->get_const_values()),
            as_cuda_type(source->get_coo()->get_const_row_idxs()),
            as_cuda_type(nnz_per_row.get_data()));

        coo_nnz =
            reduce_add_array(exec, num_rows, nnz_per_row.get_const_data());
    }

    *result = ell_nnz + coo_nnz;
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_HYBRID_COUNT_NONZEROS_KERNEL);


}  // namespace hybrid
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
