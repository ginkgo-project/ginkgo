// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
#include "hip/components/searching.hip.hpp"
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


constexpr int default_block_size = 512;


// subwarp sizes for add_candidates kernels
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/cuda_hip/factorization/par_ilut_spgeam_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void add_candidates(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* lu,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Csr<ValueType, IndexType>* u,
                    matrix::Csr<ValueType, IndexType>* l_new,
                    matrix::Csr<ValueType, IndexType>* u_new)
{
    auto num_rows = static_cast<IndexType>(lu->get_size()[0]);
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
    if (num_blocks > 0) {
        // count non-zeros per row
        kernel::tri_spgeam_nnz<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                lu_row_ptrs, lu_col_idxs, a_row_ptrs, a_col_idxs,
                l_new_row_ptrs, u_new_row_ptrs, num_rows);
    }

    // build row ptrs
    components::prefix_sum_nonnegative(exec, l_new_row_ptrs, num_rows + 1);
    components::prefix_sum_nonnegative(exec, u_new_row_ptrs, num_rows + 1);

    // resize output arrays
    auto l_new_nnz = exec->copy_val_to_host(l_new_row_ptrs + num_rows);
    auto u_new_nnz = exec->copy_val_to_host(u_new_row_ptrs + num_rows);
    l_new_builder.get_col_idx_array().resize_and_reset(l_new_nnz);
    l_new_builder.get_value_array().resize_and_reset(l_new_nnz);
    u_new_builder.get_col_idx_array().resize_and_reset(u_new_nnz);
    u_new_builder.get_value_array().resize_and_reset(u_new_nnz);

    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();
    auto u_new_col_idxs = u_new->get_col_idxs();
    auto u_new_vals = u_new->get_values();

    if (num_blocks > 0) {
        // fill columns and values
        kernel::tri_spgeam_init<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                lu_row_ptrs, lu_col_idxs, as_device_type(lu_vals), a_row_ptrs,
                a_col_idxs, as_device_type(a_vals), l_row_ptrs, l_col_idxs,
                as_device_type(l_vals), u_row_ptrs, u_col_idxs,
                as_device_type(u_vals), l_new_row_ptrs, l_new_col_idxs,
                as_device_type(l_new_vals), u_new_row_ptrs, u_new_col_idxs,
                as_device_type(u_new_vals), num_rows);
    }
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_add_candidates, add_candidates);


}  // namespace


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* lu,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Csr<ValueType, IndexType>* u,
                    matrix::Csr<ValueType, IndexType>* l_new,
                    matrix::Csr<ValueType, IndexType>* u_new)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        lu->get_num_stored_elements() + a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_add_candidates(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, lu, a, l, u, l_new,
        u_new);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_ADD_CANDIDATES_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
