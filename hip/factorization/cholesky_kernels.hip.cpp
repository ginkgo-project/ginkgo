// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/factorization/cholesky_kernels.hpp"


#include <algorithm>
#include <memory>


#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>


#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/components/format_conversion_kernels.hpp"
#include "core/factorization/elimination_forest.hpp"
#include "core/factorization/lu_kernels.hpp"
#include "core/matrix/csr_lookup.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/thrust.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/intrinsics.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/syncfree.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Cholesky namespace.
 *
 * @ingroup factor
 */
namespace cholesky {


constexpr int default_block_size = 512;


#include "common/cuda_hip/factorization/cholesky_kernels.hpp.inc"


template <typename ValueType, typename IndexType>
void symbolic_count(std::shared_ptr<const DefaultExecutor> exec,
                    const matrix::Csr<ValueType, IndexType>* mtx,
                    const factorization::elimination_forest<IndexType>& forest,
                    IndexType* row_nnz, array<IndexType>& tmp_storage)
{
    const auto num_rows = static_cast<IndexType>(mtx->get_size()[0]);
    if (num_rows == 0) {
        return;
    }
    const auto mtx_nnz = static_cast<IndexType>(mtx->get_num_stored_elements());
    tmp_storage.resize_and_reset(mtx_nnz + num_rows);
    const auto postorder_cols = tmp_storage.get_data();
    const auto lower_ends = postorder_cols + mtx_nnz;
    const auto row_ptrs = mtx->get_const_row_ptrs();
    const auto cols = mtx->get_const_col_idxs();
    const auto inv_postorder = forest.inv_postorder.get_const_data();
    const auto postorder_parent = forest.postorder_parents.get_const_data();
    // transform col indices to postorder indices
    {
        const auto num_blocks = ceildiv(num_rows, default_block_size);
        kernel::build_postorder_cols<<<num_blocks, default_block_size, 0,
                                       exec->get_stream()>>>(
            num_rows, cols, row_ptrs, inv_postorder, postorder_cols,
            lower_ends);
    }
    // sort postorder_cols inside rows
    {
        const auto handle = exec->get_hipsparse_handle();
        auto descr = hipsparse::create_mat_descr();
        array<IndexType> permutation_array(exec, mtx_nnz);
        auto permutation = permutation_array.get_data();
        components::fill_seq_array(exec, permutation, mtx_nnz);
        size_type buffer_size{};
        hipsparse::csrsort_buffer_size(handle, num_rows, num_rows, mtx_nnz,
                                       row_ptrs, postorder_cols, buffer_size);
        array<char> buffer_array{exec, buffer_size};
        auto buffer = buffer_array.get_data();
        hipsparse::csrsort(handle, num_rows, num_rows, mtx_nnz, descr, row_ptrs,
                           postorder_cols, permutation, buffer);
        hipsparse::destroy(descr);
    }
    // count nonzeros per row of L
    {
        const auto num_blocks =
            ceildiv(num_rows, default_block_size / config::warp_size);
        kernel::symbolic_count<config::warp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                num_rows, row_ptrs, lower_ends, inv_postorder, postorder_cols,
                postorder_parent, row_nnz);
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_CHOLESKY_SYMBOLIC_COUNT);


}  // namespace cholesky
}  // namespace hip
}  // namespace kernels
}  // namespace gko
