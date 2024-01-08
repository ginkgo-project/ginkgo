// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/par_ilut_kernels.hpp"


#include <CL/sycl.hpp>


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
#include "dpcpp/base/config.hpp"
#include "dpcpp/base/dim3.dp.hpp"
#include "dpcpp/components/cooperative_groups.dp.hpp"
#include "dpcpp/components/intrinsics.dp.hpp"
#include "dpcpp/components/thread_ids.dp.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr int default_block_size = 256;


// subwarp sizes for filter kernels
using compiled_kernels = syn::value_list<int, 1, 16, 32>;


#include "dpcpp/factorization/par_ilut_filter_kernels.hpp.inc"


namespace {


template <int subgroup_size, typename ValueType, typename IndexType>
void threshold_filter(syn::value_list<int, subgroup_size>,
                      std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType>* m_out,
                      matrix::Coo<ValueType, IndexType>* m_out_coo, bool lower)
{
    auto old_row_ptrs = a->get_const_row_ptrs();
    auto old_col_idxs = a->get_const_col_idxs();
    auto old_vals = a->get_const_values();
    // compute nnz for each row
    auto num_rows = static_cast<IndexType>(a->get_size()[0]);
    auto block_size = default_block_size / subgroup_size;
    auto num_blocks = ceildiv(num_rows, block_size);
    auto new_row_ptrs = m_out->get_row_ptrs();
    kernel::threshold_filter_nnz<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), old_row_ptrs,
        old_vals, num_rows, threshold, new_row_ptrs, lower);

    // build row pointers
    components::prefix_sum_nonnegative(exec, new_row_ptrs, num_rows + 1);

    // build matrix
    auto new_nnz = exec->copy_val_to_host(new_row_ptrs + num_rows);
    // resize arrays and update aliases
    matrix::CsrBuilder<ValueType, IndexType> builder{m_out};
    builder.get_col_idx_array().resize_and_reset(new_nnz);
    builder.get_value_array().resize_and_reset(new_nnz);
    auto new_col_idxs = m_out->get_col_idxs();
    auto new_vals = m_out->get_values();
    IndexType* new_row_idxs{};
    if (m_out_coo) {
        matrix::CooBuilder<ValueType, IndexType> coo_builder{m_out_coo};
        coo_builder.get_row_idx_array().resize_and_reset(new_nnz);
        coo_builder.get_col_idx_array() =
            array<IndexType>::view(exec, new_nnz, new_col_idxs);
        coo_builder.get_value_array() =
            array<ValueType>::view(exec, new_nnz, new_vals);
        new_row_idxs = m_out_coo->get_row_idxs();
    }
    kernel::threshold_filter<subgroup_size>(
        num_blocks, default_block_size, 0, exec->get_queue(), old_row_ptrs,
        old_col_idxs, old_vals, num_rows, threshold, new_row_ptrs, new_row_idxs,
        new_col_idxs, new_vals, lower);
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_threshold_filter, threshold_filter);


}  // namespace

template <typename ValueType, typename IndexType>
void threshold_filter(std::shared_ptr<const DefaultExecutor> exec,
                      const matrix::Csr<ValueType, IndexType>* a,
                      remove_complex<ValueType> threshold,
                      matrix::Csr<ValueType, IndexType>* m_out,
                      matrix::Coo<ValueType, IndexType>* m_out_coo, bool lower)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz = a->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_threshold_filter(
        compiled_kernels(),
        [&](int compiled_subgroup_size) {
            return total_nnz_per_row <= compiled_subgroup_size ||
                   compiled_subgroup_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, threshold, m_out,
        m_out_coo, lower);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
