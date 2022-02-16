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

#include "core/matrix/bccoo_kernels.hpp"


#include <hip/hip_runtime.h>


#include <ginkgo/core/base/exception_helpers.hpp>
#include <ginkgo/core/base/math.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>
#include <ginkgo/core/matrix/dense.hpp>


#include "core/components/fill_array_kernels.hpp"
#include "core/matrix/dense_kernels.hpp"
#include "hip/base/config.hip.hpp"
#include "hip/base/hipsparse_bindings.hip.hpp"
#include "hip/base/math.hip.hpp"
#include "hip/base/types.hip.hpp"
#include "hip/components/atomic.hip.hpp"
#include "hip/components/cooperative_groups.hip.hpp"
#include "hip/components/format_conversion.hip.hpp"
#include "hip/components/segment_scan.hip.hpp"
#include "hip/components/thread_ids.hip.hpp"


namespace gko {
namespace kernels {
/**
 * @brief The HIP namespace.
 *
 * @ingroup hip
 */
namespace hip {
/**
 * @brief The Bccoordinate matrix format namespace.
 *
 * @ingroup bccoo
 */
namespace bccoo {


constexpr int default_block_size = 512;
constexpr int warps_in_block = 4;
constexpr int spmv_block_size = warps_in_block * config::warp_size;


void get_default_block_size(std::shared_ptr<const HipExecutor> exec,
                            size_type* block_size) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void spmv(std::shared_ptr<const HipExecutor> exec,
          const matrix::Bccoo<ValueType, IndexType>* a,
          const matrix::Dense<ValueType>* b,
          matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv(std::shared_ptr<const HipExecutor> exec,
                   const matrix::Dense<ValueType>* alpha,
                   const matrix::Bccoo<ValueType, IndexType>* a,
                   const matrix::Dense<ValueType>* b,
                   const matrix::Dense<ValueType>* beta,
                   matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV_KERNEL);


template <typename ValueType, typename IndexType>
void spmv2(std::shared_ptr<const HipExecutor> exec,
           const matrix::Bccoo<ValueType, IndexType>* a,
           const matrix::Dense<ValueType>* b,
           matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_BCCOO_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void advanced_spmv2(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Dense<ValueType>* alpha,
                    const matrix::Bccoo<ValueType, IndexType>* a,
                    const matrix::Dense<ValueType>* b,
                    matrix::Dense<ValueType>* c) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_ADVANCED_SPMV2_KERNEL);


template <typename ValueType, typename IndexType>
void mem_size_bccoo(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::bccoo
                    : compression commpress_res,
                      size_type* mem_size) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_MEM_SIZE_BCCOO_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_compression(std::shared_ptr<const HipExecutor> exec,
                            const matrix::Bccoo<ValueType, IndexType>* source,
                            matrix::Bccoo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COMPRESSION_KERNEL);

template <typename ValueType, typename IndexType>
void convert_to_next_precision(
    std::shared_ptr<const HipExecutor> exec,
    const matrix::Bccoo<ValueType, IndexType>* source,
    matrix::Bccoo<next_precision<ValueType>, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_NEXT_PRECISION_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_coo(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Coo<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_COO_KERNEL);


template <typename IndexType>
void convert_row_idxs_to_ptrs(std::shared_ptr<const HipExecutor> exec,
                              const IndexType* idxs, size_type num_nonzeros,
                              IndexType* ptrs,
                              size_type length) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void convert_to_csr(std::shared_ptr<const HipExecutor> exec,
                    const matrix::Bccoo<ValueType, IndexType>* source,
                    matrix::Csr<ValueType, IndexType>* result)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_CSR_KERNEL);


template <typename ValueType, typename IndexType>
void convert_to_dense(std::shared_ptr<const HipExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      matrix::Dense<ValueType>* result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_CONVERT_TO_DENSE_KERNEL);


template <typename ValueType, typename IndexType>
void extract_diagonal(std::shared_ptr<const HipExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* orig,
                      matrix::Diagonal<ValueType>* diag) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_EXTRACT_DIAGONAL_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute_inplace(std::shared_ptr<const HipExecutor> exec,
                              matrix::Bccoo<ValueType, IndexType>* matrix)
    GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_INPLACE_KERNEL);


template <typename ValueType, typename IndexType>
void compute_absolute(std::shared_ptr<const HipExecutor> exec,
                      const matrix::Bccoo<ValueType, IndexType>* source,
                      remove_complex<matrix::Bccoo<ValueType, IndexType>>*
                          result) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_BCCOO_COMPUTE_ABSOLUTE_KERNEL);


}  // namespace bccoo
}  // namespace hip
}  // namespace kernels
}  // namespace gko
