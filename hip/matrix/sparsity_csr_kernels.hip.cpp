/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2022, the Ginkgo authors
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

#include "core/matrix/sparsity_csr_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/synthesizer/implementation_selection.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/reduction.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"
#include "hip/components/uninitialized_array.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Compressed sparse row matrix format namespace.
 *
 * @ingroup sparsity
 */
namespace sparsity_csr {


constexpr int classical_overweight = 32;
constexpr int spmv_block_size = 256;
constexpr int warps_in_block = 4;


using classical_kernels = syn::value_list<int, 2>;


#include "common/cuda_hip/matrix/sparsity_csr_kernels.hpp.inc"


namespace host_kernel {


template <int subwarp_size, typename ValueType, typename IndexType>
void classical_spmv(syn::value_list<int, subwarp_size>,
                    std::shared_ptr<const HipExecutor> exec,
                    const matrix::SparsityCsr<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c,
                    const matrix::Dense<ValueType>* alpha = nullptr,
                    const matrix::Dense<ValueType>* beta = nullptr)
{
    const auto nwarps = exec->get_num_warps_per_sm() *
                        exec->get_num_multiprocessor() * classical_overweight;
    const auto gridx =
        std::min(ceildiv(a->get_size()[0], spmv_block_size / subwarp_size),
                 int64(nwarps / warps_in_block));
    const dim3 grid(gridx, b->get_size()[1]);
    const dim3 block(spmv_block_size);

    if (alpha == nullptr && beta == nullptr) {
        kernel::abstract_classical_spmv<subwarp_size><<<grid, block, 0, 0>>>(
            a->get_size()[0], as_hip_type(a->get_const_value()),
            a->get_const_col_idxs(), as_hip_type(a->get_const_row_ptrs()),
            as_hip_type(b->get_const_values()), b->get_stride(),
            as_hip_type(c->get_values()), c->get_stride());

    } else if (alpha != nullptr && beta != nullptr) {
        kernel::abstract_classical_spmv<subwarp_size><<<grid, block, 0, 0>>>(
            a->get_size()[0], as_hip_type(alpha->get_const_values()),
            as_hip_type(a->get_const_value()), a->get_const_col_idxs(),
            as_hip_type(a->get_const_row_ptrs()),
            as_hip_type(b->get_const_values()), b->get_stride(),
            as_hip_type(beta->get_const_values()), as_hip_type(c->get_values()),
            c->get_stride());
    } else {
        GKO_KERNEL_NOT_FOUND;
    }
}

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_classical_spmv, classical_spmv);


}  // namespace host_kernel

template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::SparsityCsr<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b, matrix::Dense<ValueType>* c)
{
    host_kernel::select_classical_spmv(
        classical_kernels(), [](int compiled_info) { return true; },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::SparsityCsr<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c)
{
    host_kernel::select_classical_spmv(
        classical_kernels(), [](int compiled_info) { return true; },
        syn::value_list<int>(), syn::type_list<>(), exec, a, b, c, alpha, beta);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void fill_in_dense(std::shared_ptr<const DefaultExecutor> exec,
                   const matrix::SparsityCsr<ValueType, IndexType>* input,
                   matrix::Dense<ValueType>* output) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_FILL_IN_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void count_num_diagonal_elements(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* matrix,
    size_type* num_diagonal_elements) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_COUNT_NUM_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void remove_diagonal_elements(
    std::shared_ptr<const HipExecutor> exec, const IndexType* row_ptrs,
    const IndexType* col_idxs,
    matrix::SparsityCsr<ValueType, IndexType>* matrix) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_REMOVE_DIAGONAL_ELEMENTS_KERNEL);


template <typename ValueType, typename IndexType>
void transpose(std::shared_ptr<const HipExecutor> exec,
               const matrix::SparsityCsr<ValueType, IndexType>* orig,
               matrix::SparsityCsr<ValueType, IndexType>* trans)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_TRANSPOSE_KERNEL);


template <typename ValueType, typename IndexType>
void sort_by_column_index(std::shared_ptr<const HipExecutor> exec,
                          matrix::SparsityCsr<ValueType, IndexType>* to_sort)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_SORT_BY_COLUMN_INDEX);


template <typename ValueType, typename IndexType>
void is_sorted_by_column_index(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::SparsityCsr<ValueType, IndexType>* to_check,
    bool* is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SPARSITY_CSR_IS_SORTED_BY_COLUMN_INDEX);


}  // namespace sparsity_csr
}  // namespace hip
}  // namespace kernels
}  // namespace gko
