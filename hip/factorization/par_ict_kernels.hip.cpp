// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ict_kernels.hpp"


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
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/memory.hip.hpp"
#include "hip/components/merging.hip.hpp"
#include "hip/components/prefix_sum.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/searching.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The parallel ICT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ict_factorization {


constexpr int default_block_size = 512;


// subwarp sizes for all warp-parallel kernels (filter, add_candidates)
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/cuda_hip/factorization/par_ict_spgeam_kernels.hpp.inc"
#include "common/cuda_hip/factorization/par_ict_sweep_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void add_candidates(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* llh,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    matrix::Csr<ValueType, IndexType>* l_new)
{
    auto num_rows = static_cast<IndexType>(llh->get_size()[0]);
    auto subwarps_per_block = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, subwarps_per_block);
    matrix::CsrBuilder<ValueType, IndexType> l_new_builder(l_new);
    auto llh_row_ptrs = llh->get_const_row_ptrs();
    auto llh_col_idxs = llh->get_const_col_idxs();
    auto llh_vals = llh->get_const_values();
    auto a_row_ptrs = a->get_const_row_ptrs();
    auto a_col_idxs = a->get_const_col_idxs();
    auto a_vals = a->get_const_values();
    auto l_row_ptrs = l->get_const_row_ptrs();
    auto l_col_idxs = l->get_const_col_idxs();
    auto l_vals = l->get_const_values();
    auto l_new_row_ptrs = l_new->get_row_ptrs();
    // count non-zeros per row
    if (num_blocks > 0) {
        kernel::ict_tri_spgeam_nnz<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                llh_row_ptrs, llh_col_idxs, a_row_ptrs, a_col_idxs,
                l_new_row_ptrs, num_rows);
    }

    // build row ptrs
    components::prefix_sum_nonnegative(exec, l_new_row_ptrs, num_rows + 1);

    // resize output arrays
    auto l_new_nnz = exec->copy_val_to_host(l_new_row_ptrs + num_rows);
    l_new_builder.get_col_idx_array().resize_and_reset(l_new_nnz);
    l_new_builder.get_value_array().resize_and_reset(l_new_nnz);

    auto l_new_col_idxs = l_new->get_col_idxs();
    auto l_new_vals = l_new->get_values();

    // fill columns and values
    if (num_blocks > 0) {
        kernel::ict_tri_spgeam_init<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                llh_row_ptrs, llh_col_idxs, as_device_type(llh_vals),
                a_row_ptrs, a_col_idxs, as_device_type(a_vals), l_row_ptrs,
                l_col_idxs, as_device_type(l_vals), l_new_row_ptrs,
                l_new_col_idxs, as_device_type(l_new_vals), num_rows);
    }
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_add_candidates, add_candidates);


template <int subwarp_size, typename ValueType, typename IndexType>
void compute_factor(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* a,
                    matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Coo<ValueType, IndexType>* l_coo)
{
    auto total_nnz = static_cast<IndexType>(l->get_num_stored_elements());
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(total_nnz, block_size);
    if (num_blocks > 0) {
        kernel::ict_sweep<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                a->get_const_row_ptrs(), a->get_const_col_idxs(),
                as_device_type(a->get_const_values()), l->get_const_row_ptrs(),
                l_coo->get_const_row_idxs(), l->get_const_col_idxs(),
                as_device_type(l->get_values()),
                static_cast<IndexType>(l->get_num_stored_elements()));
    }
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_factor, compute_factor);


}  // namespace


template <typename ValueType, typename IndexType>
void add_candidates(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* llh,
                    const matrix::Csr<ValueType, IndexType>* a,
                    const matrix::Csr<ValueType, IndexType>* l,
                    matrix::Csr<ValueType, IndexType>* l_new)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        llh->get_num_stored_elements() + a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_add_candidates(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, llh, a, l, l_new);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ICT_ADD_CANDIDATES_KERNEL);


template <typename ValueType, typename IndexType>
void compute_factor(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* a,
                    matrix::Csr<ValueType, IndexType>* l,
                    const matrix::Coo<ValueType, IndexType>* l_coo)
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
}  // namespace hip
}  // namespace kernels
}  // namespace gko
