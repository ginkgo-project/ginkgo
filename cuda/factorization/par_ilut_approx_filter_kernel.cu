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


#include <algorithm>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/matrix/coo.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/factorization/par_ilut_kernels.hpp"
#include "core/matrix/coo_builder.hpp"
#include "core/matrix/csr_builder.hpp"
#include "core/matrix/csr_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/base/config.hpp"
#include "cuda/base/math.hpp"
#include "cuda/base/types.hpp"
#include "cuda/components/atomic.cuh"
#include "cuda/components/cooperative_groups.cuh"
#include "cuda/components/intrinsics.cuh"
#include "cuda/components/prefix_sum.cuh"
#include "cuda/components/sorting.cuh"
#include "cuda/components/thread_ids.cuh"
#include "cuda/factorization/par_ilut_select_common.cuh"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


// subwarp sizes for filter kernels
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/factorization/par_ilut_filter_kernels.hpp.inc"
#include "common/factorization/par_ilut_select_kernels.hpp.inc"


template <int subwarp_size, typename ValueType, typename IndexType>
void threshold_filter_approx(syn::value_list<int, subwarp_size>,
                             std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Csr<ValueType, IndexType> *m,
                             IndexType rank, Array<ValueType> *tmp,
                             remove_complex<ValueType> *threshold,
                             matrix::Csr<ValueType, IndexType> *m_out,
                             matrix::Coo<ValueType, IndexType> *m_out_coo)
{
    auto values = m->get_const_values();
    IndexType size = m->get_num_stored_elements();
    using AbsType = remove_complex<ValueType>;
    constexpr auto bucket_count = kernel::searchtree_width;
    auto max_num_threads = ceildiv(size, items_per_thread);
    auto max_num_blocks = ceildiv(max_num_threads, default_block_size);

    size_type tmp_size_totals =
        ceildiv((bucket_count + 1) * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_partials = ceildiv(
        bucket_count * max_num_blocks * sizeof(IndexType), sizeof(ValueType));
    size_type tmp_size_oracles =
        ceildiv(size * sizeof(unsigned char), sizeof(ValueType));
    size_type tmp_size_tree =
        ceildiv(kernel::searchtree_size * sizeof(AbsType), sizeof(ValueType));
    size_type tmp_size =
        tmp_size_totals + tmp_size_partials + tmp_size_oracles + tmp_size_tree;
    tmp->resize_and_reset(tmp_size);

    auto total_counts = reinterpret_cast<IndexType *>(tmp->get_data());
    auto partial_counts =
        reinterpret_cast<IndexType *>(tmp->get_data() + tmp_size_totals);
    auto oracles = reinterpret_cast<unsigned char *>(
        tmp->get_data() + tmp_size_totals + tmp_size_partials);
    auto tree =
        reinterpret_cast<AbsType *>(tmp->get_data() + tmp_size_totals +
                                    tmp_size_partials + tmp_size_oracles);

    ssss_count(values, size, tree, oracles, partial_counts, total_counts);

    // determine bucket with correct rank
    auto bucket = static_cast<unsigned char>(
        ssss_find_bucket(exec, total_counts, rank).idx);
    exec->get_master()->copy_from(exec.get(), 1,
                                  tree + kernel::searchtree_inner_size + bucket,
                                  threshold);
    // we implicitly set the first splitter to -inf, but 0 works as well
    if (bucket == 0) {
        *threshold = zero<AbsType>();
    }

    // filter the elements
    auto old_row_ptrs = m->get_const_row_ptrs();
    auto old_col_idxs = m->get_const_col_idxs();
    auto old_vals = m->get_const_values();
    // compute nnz for each row
    auto num_rows = static_cast<IndexType>(m->get_size()[0]);
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(num_rows, block_size);
    auto new_row_ptrs = m_out->get_row_ptrs();
    kernel::bucket_filter_nnz<subwarp_size><<<num_blocks, default_block_size>>>(
        old_row_ptrs, oracles, num_rows, bucket, new_row_ptrs);

    // build row pointers
    components::prefix_sum(exec, new_row_ptrs, num_rows + 1);

    // build matrix
    IndexType new_nnz{};
    exec->get_master()->copy_from(exec.get(), 1, new_row_ptrs + num_rows,
                                  &new_nnz);
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
    kernel::bucket_filter<subwarp_size><<<num_blocks, default_block_size>>>(
        old_row_ptrs, old_col_idxs, as_cuda_type(old_vals), oracles, num_rows,
        bucket, new_row_ptrs, new_row_idxs, new_col_idxs,
        as_cuda_type(new_vals));
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_threshold_filter_approx,
                                    threshold_filter_approx);


template <typename ValueType, typename IndexType>
void threshold_filter_approx(std::shared_ptr<const DefaultExecutor> exec,
                             const matrix::Csr<ValueType, IndexType> *m,
                             IndexType rank, Array<ValueType> &tmp,
                             remove_complex<ValueType> &threshold,
                             matrix::Csr<ValueType, IndexType> *m_out,
                             matrix::Coo<ValueType, IndexType> *m_out_coo)
{
    auto num_rows = m->get_size()[0];
    auto total_nnz = m->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_threshold_filter_approx(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, m, rank, &tmp,
        &threshold, m_out, m_out_coo);
}


GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_THRESHOLD_FILTER_APPROX_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
