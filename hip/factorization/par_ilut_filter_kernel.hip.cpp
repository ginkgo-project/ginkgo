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

#include "core/factorization/par_ilut_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr auto default_block_size = 512;


// subwarp sizes for filter kernels
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/factorization/par_ilut_filter_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter(syn::value_list<int, subwarp_size>,
                      std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType> *m_out,
                      matrix::Coo<ValueType, IndexType> *m_out_coo, bool lower)
{
    auto old_row_ptrs = a->get_const_row_ptrs();
    auto old_col_idxs = a->get_const_col_idxs();
    auto old_vals = a->get_const_values();
    // compute nnz for each row
    auto num_rows = static_cast<IndexType>(a->get_size()[0]);
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, block_size);
    auto new_row_ptrs = m_out->get_row_ptrs();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel::threshold_filter_nnz<subwarp_size>),
        dim3(num_blocks), dim3(default_block_size), 0, 0, old_row_ptrs,
        as_hip_type(old_vals), num_rows, threshold, new_row_ptrs, lower);

    // build row pointers
    components::prefix_sum(exec, new_row_ptrs, num_rows + 1);

    // build matrix
    auto new_nnz = exec->copy_val_to_host(new_row_ptrs + num_rows);
    // resize arrays and update aliases
    matrix::CsrBuilder<ValueType, IndexType> builder{m_out};
    builder.get_col_idx_array().resize_and_reset(new_nnz);
    builder.get_value_array().resize_and_reset(new_nnz);
    auto new_col_idxs = m_out->get_col_idxs();
    auto new_vals = m_out->get_values();
    IndexType *new_row_idxs{};
    if (m_out_coo) {
        matrix::CooBuilder<ValueType, IndexType> coo_builder{m_out_coo};
        coo_builder.get_row_idx_array().resize_and_reset(new_nnz);
        coo_builder.get_col_idx_array() =
            Array<IndexType>::view(exec, new_nnz, new_col_idxs);
        coo_builder.get_value_array() =
            Array<ValueType>::view(exec, new_nnz, new_vals);
        new_row_idxs = m_out_coo->get_row_idxs();
    }
    hipLaunchKernelGGL(HIP_KERNEL_NAME(kernel::threshold_filter<subwarp_size>),
                       dim3(num_blocks), dim3(default_block_size), 0, 0,
                       old_row_ptrs, old_col_idxs, as_hip_type(old_vals),
                       num_rows, threshold, new_row_ptrs, new_row_idxs,
                       new_col_idxs, as_hip_type(new_vals), lower);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_threshold_filter, threshold_filter);


}  // namespace

template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType> *a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType> *m_out,
                      matrix::Coo<ValueType, IndexType> *m_out_coo, bool lower)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz = a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_threshold_filter(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, threshold, m_out,
        m_out_coo, lower);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko