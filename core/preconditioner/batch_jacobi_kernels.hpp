/*******************************<GINKGO LICENSE>******************************
Copyright (c) 2017-2023, the Ginkgo authors
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

#ifndef GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
#define GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_

#include <ginkgo/core/matrix/batch_csr.hpp>
#include <ginkgo/core/matrix/batch_dense.hpp>
#include <ginkgo/core/matrix/batch_ell.hpp>
#include <ginkgo/core/preconditioner/batch_jacobi.hpp>


#include "core/base/kernel_declaration.hpp"


namespace gko {
namespace kernels {


/**
 * @fn batch_jacobi_apply
 *
 * This kernel builds a Jacobi preconditioner for each matrix in
 * the input batch of matrices and applies them to the corresponding vectors
 * in the input vector batches.
 *
 * These functions are mostly meant only for experimentation and testing.
 *
 * @param exec  The executor on which to run the kernel.
 * @param a  The batch of matrices for which to build the preconditioner.
 * @param b  The batch of input (RHS) vectors.
 * @param x  The batch of output (solution) vectors.
 */
#define GKO_DECLARE_BATCH_SCALAR_JACOBI_APPLY_KERNEL(_type)              \
    void batch_jacobi_apply(std::shared_ptr<const DefaultExecutor> exec, \
                            const matrix::BatchCsr<_type>* a,            \
                            const matrix::BatchDense<_type>* b,          \
                            matrix::BatchDense<_type>* x)

#define GKO_DECLARE_BATCH_SCALAR_JACOBI_ELL_APPLY_KERNEL(_type)          \
    void batch_jacobi_apply(std::shared_ptr<const DefaultExecutor> exec, \
                            const matrix::BatchEll<_type>* a,            \
                            const matrix::BatchDense<_type>* b,          \
                            matrix::BatchDense<_type>* x)

#define GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL(ValueType,     \
                                                              IndexType)     \
    void extract_common_blocks_pattern(                                      \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::Csr<ValueType, IndexType>* first_sys_csr,              \
        const uint32 max_block_size, const size_type num_blocks,             \
        const preconditioner::batched_blocks_storage_scheme& storage_scheme, \
        const IndexType* block_pointers, IndexType* blocks_pattern)


#define GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL(ValueType, IndexType)  \
    void compute_block_jacobi(                                               \
        std::shared_ptr<const DefaultExecutor> exec,                         \
        const matrix::BatchCsr<ValueType, IndexType>* sys_csr,               \
        const size_type num_blocks, const uint32 max_block_size,             \
        const preconditioner::batched_blocks_storage_scheme& storage_scheme, \
        const IndexType* block_pointers, const IndexType* blocks_pattern,    \
        ValueType* blocks)

#define GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL(ValueType, IndexType) \
    void batch_jacobi_apply(                                        \
        std::shared_ptr<const DefaultExecutor> exec,                \
        const matrix::BatchCsr<ValueType, IndexType>* sys_mat,      \
        const size_type num_blocks, const uint32 max_block_size,    \
        const ValueType* blocks_array, const IndexType* block_ptrs, \
        const matrix::BatchDense<ValueType>* r,                     \
        matrix::BatchDense<ValueType>* z)

#define GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL(ValueType, IndexType) \
    void batch_jacobi_apply(                                            \
        std::shared_ptr<const DefaultExecutor> exec,                    \
        const matrix::BatchEll<ValueType, IndexType>* sys_mat,          \
        const size_type num_blocks, const uint32 max_block_size,        \
        const ValueType* blocks_array, const IndexType* block_ptrs,     \
        const matrix::BatchDense<ValueType>* r,                         \
        matrix::BatchDense<ValueType>* z)

#define GKO_DECLARE_ALL_AS_TEMPLATES                                     \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_SCALAR_JACOBI_ELL_APPLY_KERNEL(ValueType);         \
    template <typename ValueType>                                        \
    GKO_DECLARE_BATCH_SCALAR_JACOBI_APPLY_KERNEL(ValueType);             \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_EXTRACT_PATTERN_KERNEL(ValueType,     \
                                                          IndexType);    \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_BLOCK_JACOBI_COMPUTE_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_JACOBI_ELL_APPLY_KERNEL(ValueType, IndexType);     \
    template <typename ValueType, typename IndexType>                    \
    GKO_DECLARE_BATCH_JACOBI_APPLY_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(batch_jacobi,
                                        GKO_DECLARE_ALL_AS_TEMPLATES);
#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_PRECONDITIONER_BATCH_JACOBI_KERNELS_HPP_
