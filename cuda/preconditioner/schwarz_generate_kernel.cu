/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2021, the Ginkgo authors
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

#include "core/preconditioner/schwarz_kernels.hpp"


#include <ginkgo/config.hpp>
#include <ginkgo/core/base/exception_helpers.hpp>


#include "core/components/fill_array.hpp"
#include "core/synthesizer/implementation_selection.hpp"
#include "cuda/preconditioner/schwarz_common.hpp"


namespace gko {
namespace kernels {
namespace cuda {
/**
 * @brief The Schwarz preconditioner namespace.
 * @ref Schwarz
 * @ingroup schwarz
 */
namespace schwarz {


template <int warps_per_block, int max_block_size, typename ValueType,
          typename IndexType>
void generate(syn::value_list<int, max_block_size>,
              const matrix::Csr<ValueType, IndexType>* mtx,
              remove_complex<ValueType> accuracy, ValueType* block_data,
              const preconditioner::block_interleaved_storage_scheme<IndexType>&
                  storage_scheme,
              remove_complex<ValueType>* conditioning,
              precision_reduction* block_precisions,
              const IndexType* block_ptrs, size_type num_blocks);

GKO_ENABLE_IMPLEMENTATION_SELECTION(select_generate, generate);


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CudaExecutor> exec,
              const matrix::Csr<ValueType, IndexType>* system_matrix,
              size_type num_blocks, uint32 max_block_size,
              remove_complex<ValueType> accuracy,
              const preconditioner::block_interleaved_storage_scheme<IndexType>&
                  storage_scheme,
              Array<remove_complex<ValueType>>& conditioning,
              Array<precision_reduction>& block_precisions,
              const Array<IndexType>& block_pointers,
              Array<ValueType>& blocks) GKO_NOT_IMPLEMENTED;
//{
// TODO (script:schwarz): change the code imported from preconditioner/jacobi if
// needed
//    components::fill_array(exec, blocks.get_data(), blocks.get_num_elems(),
//                           zero<ValueType>());
//    select_generate(
//        compiled_kernels(),
//        [&](int compiled_block_size) {
//            return max_block_size <= compiled_block_size;
//        },
//        syn::value_list<int, config::min_warps_per_block>(),
//        syn::type_list<>(), system_matrix, accuracy, blocks.get_data(),
//        storage_scheme, conditioning.get_data(), block_precisions.get_data(),
//        block_pointers.get_const_data(), num_blocks);
//}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_SCHWARZ_GENERATE_KERNEL);


}  // namespace schwarz
}  // namespace cuda
}  // namespace kernels
}  // namespace gko
