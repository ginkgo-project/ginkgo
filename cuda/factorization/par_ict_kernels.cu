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

#include "core/factorization/par_ict_kernels.hpp"


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
#include "cuda/base/math.hpp"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/reduction.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ICT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ict_factorization {


constexpr auto default_block_size = 512;


// subwarp sizes for all warp-parallel kernels (filter, add_candidates)
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/factorization/par_ict_spgeam_kernels.hpp.inc"
#include "common/factorization/par_ict_sweep_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void add_candidates(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *llt,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    matrix::Csr<ValueType, IndexType> *l_new)
{
    auto num_rows = static_cast<IndexType>(llt->get_size()[0]);
    auto subwarps_per_block = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, subwarps_per_block);
    matrix::CsrBuilder<ValueType, IndexType> l_new_builder(l_new);
    auto llt_row_ptrs = llt->get_const_row_ptrs();
    auto llt_col_idxs = llt->get_const_col_idxs();
    auto llt_vals = llt->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    // count non-zeros per row
    kernel::ict_tri_spgeam_nnz<subwarp_size>
        <<<num_blocks, default_block_size>>>(llt_row_ptrs, llt_col_idxs,
                                             a_row_ptrs, a_col_idxs,
                                             l_new_row_ptrs, num_rows);

    // build row ptrs
    components::prefix_sum(exec, l_new_row_ptrs, num_rows + 1);

    // resize output arrays
    auto l_new_nnz = exec->copy_val_to_host(l_new_row_ptrs + num_rows);
    l_new_builder.get_col_idx_array().resize_and_reset(l_new_nnz);
    l_new_builder.get_value_array().resize_and_reset(l_new_nnz);

    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();

    // fill columns and values
    kernel::ict_tri_spgeam_init<subwarp_size>
        <<<num_blocks, default_block_size>>>(
            llt_row_ptrs, llt_col_idxs, as_cuda_type(llt_vals), a_row_ptrs,
            a_col_idxs, as_cuda_type(a_vals), l_row_ptrs, l_col_idxs,
            as_cuda_type(l_vals), l_new_row_ptrs, l_new_col_idxs,
            as_cuda_type(l_new_vals), num_rows);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_add_candidates, add_candidates);


template <int subwarp_size, typename ValueType, typename IndexType>
void compute_factor(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *a,
                    matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Coo<ValueType, IndexType> *l_coo)
{
    auto total_nnz = static_cast<IndexType>(l->get_num_stored_elements());
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(total_nnz, block_size);
    kernel::ict_sweep<subwarp_size><<<num_blocks, default_block_size>>>(
        a->get_const_row_ptrs(), a->get_const_col_idxs(),
        as_cuda_type(a->get_const_values()), l->get_const_row_ptrs(),
        l_coo->get_const_row_idxs(), l->get_const_col_idxs(),
        as_cuda_type(l->get_values()),
        static_cast<IndexType>(l->get_num_stored_elements()));
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factor, compute_factor);


}  // namespace


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *llt,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    matrix::Csr<ValueType, IndexType> *l_new)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        llt->get_num_stored_elements() + a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_add_candidates(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, llt, a, l, l_new);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *a,
                    matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Coo<ValueType, IndexType> *l_coo)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz = 2 * l->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_compute_factor(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, l, l_coo);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_COMPUTE_FACTOR_KERNEL);


}  // namespace par_ict_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
