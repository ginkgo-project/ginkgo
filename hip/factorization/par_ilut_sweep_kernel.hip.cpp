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
 * @brief The parallel ILUT factorization namespace.
 *
 * @ingroup factor
 */
namespace par_ilut_factorization {


constexpr int default_block_size = 512;


// subwarp sizes for all warp-parallel kernels (filter, add_candidates)
using compiled_kernels =
    syn::value_list<int, 1, 2, 4, 8, 16, 32, config::warp_size>;


#include "common/cuda_hip/factorization/par_ilut_sweep_kernels.hpp.inc"


namespace {


template <int subwarp_size, typename ValueType, typename IndexType>
void compute_l_u_factors(syn::value_list<int, subwarp_size>,
                         std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* a,
                         matrix::Csr<ValueType, IndexType>* l,
                         const matrix::Coo<ValueType, IndexType>* l_coo,
                         matrix::Csr<ValueType, IndexType>* u,
                         const matrix::Coo<ValueType, IndexType>* u_coo,
                         matrix::Csr<ValueType, IndexType>* u_csc)
{
    auto total_nnz = static_cast<IndexType>(l->get_num_stored_elements() +
                                            u->get_num_stored_elements());
    auto block_size = default_block_size / subwarp_size;
    auto num_blocks = ceildiv(total_nnz, block_size);
    if (num_blocks > 0) {
        kernel::sweep<subwarp_size>
            <<<num_blocks, default_block_size, 0, exec->get_stream()>>>(
                a->get_const_row_ptrs(), a->get_const_col_idxs(),
                as_device_type(a->get_const_values()), l->get_const_row_ptrs(),
                l_coo->get_const_row_idxs(), l->get_const_col_idxs(),
                as_device_type(l->get_values()),
                static_cast<IndexType>(l->get_num_stored_elements()),
                u_coo->get_const_row_idxs(), u_coo->get_const_col_idxs(),
                as_device_type(u->get_values()), u_csc->get_const_row_ptrs(),
                u_csc->get_const_col_idxs(),
                as_device_type(u_csc->get_values()),
                static_cast<IndexType>(u->get_num_stored_elements()));
    }
}


GKO_ENABLE_IMPLEMENTATION_SELECTION(select_compute_l_u_factors,
                                    compute_l_u_factors);


}  // namespace


template <typename ValueType, typename IndexType>
void compute_l_u_factors(std::shared_ptr<const DefaultExecutor> exec,
                         const matrix::Csr<ValueType, IndexType>* a,
                         matrix::Csr<ValueType, IndexType>* l,
                         const matrix::Coo<ValueType, IndexType>* l_coo,
                         matrix::Csr<ValueType, IndexType>* u,
                         const matrix::Coo<ValueType, IndexType>* u_coo,
                         matrix::Csr<ValueType, IndexType>* u_csc)
{
    auto num_rows = a->get_size()[0];
    auto total_nnz =
        l->get_num_stored_elements() + u->get_num_stored_elements();
    auto total_nnz_per_row = total_nnz / num_rows;
    select_compute_l_u_factors(
        compiled_kernels(),
        [&](int compiled_subwarp_size) {
            return total_nnz_per_row <= compiled_subwarp_size ||
                   compiled_subwarp_size == config::warp_size;
        },
        syn::value_list<int>(), syn::type_list<>(), exec, a, l, l_coo, u, u_coo,
        u_csc);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PAR_ILUT_COMPUTE_LU_FACTORS_KERNEL);


}  // namespace par_ilut_factorization
}  // namespace hip
}  // namespace kernels
}  // namespace gko
