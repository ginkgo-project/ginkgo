/*******************************<GINKGO LICENSE>******************************
Copyright 2017-2018

Karlsruhe Institute of Technology
Universitat Jaume I
University of Tennessee

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
******************************<GINKGO LICENSE>*******************************/

#include "core/preconditioner/block_jacobi_kernels.hpp"


#include "core/base/exception_helpers.hpp"


namespace gko {
namespace kernels {
namespace cpu {
namespace block_jacobi {


template <typename ValueType, typename IndexType>
void find_blocks(std::shared_ptr<const CpuExecutor> exec,
                 const matrix::Csr<ValueType, IndexType> *system_matrix,
                 uint32 max_block_size, size_type &num_blocks,
                 Array<IndexType> &block_pointers) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_FIND_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CpuExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type padding,
              const Array<IndexType> &block_pointers,
              Array<ValueType> &blocks) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const CpuExecutor> exec, size_type num_blocks,
           uint32 max_block_size, size_type padding,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta,
           matrix::Dense<ValueType> *x) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(std::shared_ptr<const CpuExecutor> exec, size_type num_blocks,
                  uint32 max_block_size, size_type padding,
                  const Array<IndexType> &block_pointers,
                  const Array<ValueType> &blocks,
                  const matrix::Dense<ValueType> *b,
                  matrix::Dense<ValueType> *x) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const CpuExecutor> exec,
                      size_type num_blocks,
                      const Array<IndexType> &block_pointers,
                      const Array<ValueType> &blocks, size_type block_padding,
                      ValueType *result_values,
                      size_type result_padding) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace block_jacobi


namespace adaptive_block_jacobi {


template <typename ValueType, typename IndexType>
void generate(std::shared_ptr<const CpuExecutor> exec,
              const matrix::Csr<ValueType, IndexType> *system_matrix,
              size_type num_blocks, uint32 max_block_size, size_type padding,
              Array<precision<ValueType, IndexType>> &block_precisions,
              const Array<IndexType> &block_pointers,
              Array<ValueType> &blocks) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_GENERATE_KERNEL);


template <typename ValueType, typename IndexType>
void apply(std::shared_ptr<const CpuExecutor> exec, size_type num_blocks,
           uint32 max_block_size, size_type padding,
           const Array<precision<ValueType, IndexType>> &block_precisions,
           const Array<IndexType> &block_pointers,
           const Array<ValueType> &blocks,
           const matrix::Dense<ValueType> *alpha,
           const matrix::Dense<ValueType> *b,
           const matrix::Dense<ValueType> *beta,
           matrix::Dense<ValueType> *x) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void simple_apply(
    std::shared_ptr<const CpuExecutor> exec, size_type num_blocks,
    uint32 max_block_size, size_type padding,
    const Array<precision<ValueType, IndexType>> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    const matrix::Dense<ValueType> *b,
    matrix::Dense<ValueType> *x) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_SIMPLE_APPLY_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(
    std::shared_ptr<const CpuExecutor> exec, size_type num_blocks,
    const Array<precision<ValueType, IndexType>> &block_precisions,
    const Array<IndexType> &block_pointers, const Array<ValueType> &blocks,
    size_type block_padding, ValueType *result_values,
    size_type result_padding) NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_ADAPTIVE_BLOCK_JACOBI_CONVERT_TO_DENSE_KERNEL);


}  // namespace adaptive_block_jacobi
}  // namespace cpu
}  // namespace kernels
}  // namespace gko
