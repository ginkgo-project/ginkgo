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
    syn::value_list<int, max_block_size>, size_type num_blocks,
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
            syn::type_list<>(), num_blocks, block_precisions.get_const_data(),
            block_pointers.get_const_data(), blocks.get_const_data(),
            storage_scheme, alpha->get_const_values(),
            b->get_const_values() + col, b->get_stride(), x->get_values() + col,
            x->get_stride());
    }
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_JACOBI_APPLY_KERNEL);


}  // namespace jacobi
}  // namespace hip
}  // namespace kernels
}  // namespace gko
