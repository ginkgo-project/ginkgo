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
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/merging.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/searching.cuh"
#include "cuda/components/thread_ids.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr auto default_block_size = 512;


// subwarp sizes for add_candidates kernels
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/factorization/par_ilut_spgeam_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void add_candidates(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *lu,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Csr<ValueType, IndexType> *u,
                    matrix::Csr<ValueType, IndexType> *l_new,
                    matrix::Csr<ValueType, IndexType> *u_new)
{
    auto num_rows = static_cast<IndexType>(lu->get_size()[0]);
    auto host_exec = exec->get_master();
    auto subwarps_per_block = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, subwarps_per_block);
    matrix::CsrBuilder<ValueType, IndexType> l_new_builder(l_new);
    matrix::CsrBuilder<ValueType, IndexType> u_new_builder(u_new);
    auto lu_row_ptrs = lu->get_const_row_ptrs();
    auto lu_col_idxs = lu->get_const_col_idxs();
    auto lu_vals = lu->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto u_row_ptrs = u->get_const_row_ptrs();
    auto u_col_idxs = u->get_const_col_idxs();
    auto u_vals = u->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    auto u_new_row_ptrs = u_new->get_row_ptrs();
    // count non-zeros per row
    kernel::tri_spgeam_nnz<subwarp_size><<<num_blocks, default_block_size>>>(
        lu_row_ptrs, lu_col_idxs, a_row_ptrs, a_col_idxs, l_new_row_ptrs,
        u_new_row_ptrs, num_rows);

    // build row ptrs
    components::prefix_sum(exec, l_new_row_ptrs, num_rows + 1);
    components::prefix_sum(exec, u_new_row_ptrs, num_rows + 1);

    // resize output arrays
    IndexType l_new_nnz{};
    IndexType u_new_nnz{};
    host_exec->copy_from(exec.get(), 1, l_new_row_ptrs + num_rows, &l_new_nnz);
    host_exec->copy_from(exec.get(), 1, u_new_row_ptrs + num_rows, &u_new_nnz);
    l_new_builder.get_col_idx_array().resize_and_reset(l_new_nnz);
    l_new_builder.get_value_array().resize_and_reset(l_new_nnz);
    u_new_builder.get_col_idx_array().resize_and_reset(u_new_nnz);
    u_new_builder.get_value_array().resize_and_reset(u_new_nnz);

    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();
    auto u_new_col_idxs = u_new->get_col_idxs();
    auto u_new_vals = u_new->get_values();

    // fill columns and values
    kernel::tri_spgeam_init<subwarp_size><<<num_blocks, default_block_size>>>(
        lu_row_ptrs, lu_col_idxs, as_cuda_type(lu_vals), a_row_ptrs, a_col_idxs,
        as_cuda_type(a_vals), l_row_ptrs, l_col_idxs, as_cuda_type(l_vals),
        u_row_ptrs, u_col_idxs, as_cuda_type(u_vals), l_new_row_ptrs,
        l_new_col_idxs, as_cuda_type(l_new_vals), u_new_row_ptrs,
        u_new_col_idxs, as_cuda_type(u_new_vals), num_rows);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_add_candidates, add_candidates);


}  // namespace


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType> *lu,
                    const matrix::Csr<ValueType, IndexType> *a,
                    const matrix::Csr<ValueType, IndexType> *l,
                    const matrix::Csr<ValueType, IndexType> *u,
                    matrix::Csr<ValueType, IndexType> *l_new,
                    matrix::Csr<ValueType, IndexType> *u_new)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        lu->get_num_stored_elements() + a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_add_candidates(compiled_kernels(),
                          [&](int compiled_subwarp_size) {
                              return total_nnz_per_row <=
                                         compiled_subwarp_size ||
                                     compiled_subwarp_size == config::warp_size;
                          },
                          syn::value_list<int>(), syn::type_list<>(), exec, lu,
                          a, l, u, l_new, u_new);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
