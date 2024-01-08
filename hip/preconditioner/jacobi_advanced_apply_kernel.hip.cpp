// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#include "core/preconditioner/jacobi_kernels.hpp"


#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/matrix/dense_kernels.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "hip/preconditioner/jacobi_common.hip.hpp"


namespace gko {
namespace kernels {
namespace hip {
/**
 * @brief The Jacobi preconditioner namespace.
 * @ref Jacobi
 * @ingroup jacobi
 */
namespace jacobi {


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void advanced_apply(
    syn::value_list<int, max_block_size>,
    std::shared_ptr<const DefaultExecutor> exec, size_type num_blocks,
    const precision_reduction* block_precisions,
    const IndexType* block_pointers, const ValueType* blocks,
    const preconditioner::block_interleaved_storage_scheme<IndexType>&
        storage_scheme,
    const ValueType* alpha, const ValueType* b, size_type b_stride,
    ValueType* x, size_type x_stride);

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_advanced_apply, advanced_apply);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const HipExecutor> exec, size_type num_blocks,
           uint32 max_block_size,
           const preconditioner::block_interleaved_storage_scheme<IndexType>&
               storage_scheme,
           const array<precision_reduction>& block_precisions,
           const array<IndexType>& block_pointers,
           const array<ValueType>& blocks,
           const matrix::Dense<ValueType>* alpha,
           const matrix::Dense<ValueType>* b,
           const matrix::Dense<ValueType>* beta, matrix::Dense<ValueType>* x)
{
    // TODO: write a special kernel for multiple RHS
    dense::scale(exec, beta, x);
    for (size_type col = 0; col < b->get_size()[1]; ++col) {
        select_advanced_apply(
            compiled_kernels(),
            [&](int compiled_block_size) {
                return max_block_size <= compiled_block_size;
            },
            syn::value_list<int, config::min_warps_per_block>(),
            syn::type_list<>(), exec, num_blocks,
            block_precisions.get_const_data(), block_pointers.get_const_data(),
            blocks.get_const_data(), storage_scheme, alpha->get_const_values(),
            b->get_const_values() + col, b->get_stride(), x->get_values() + col,
            x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_APPLY_KERNEL);


}  // namespace jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
