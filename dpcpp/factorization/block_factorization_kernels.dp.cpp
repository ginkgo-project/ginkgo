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

#include "core/factorization/block_factorization_kernels.hpp"


#include <algorithm>
#include <memory>


#include <ginkgo/core/base/array.hpp>


#include "core/components/prefix_sum.hpp"
#include "core/matrix/fbcsr_builder.hpp"


namespace gko {
namespace kernels {
namespace dpcpp {
/**
 * @brief The factorization namespace.
 *
 * @ingroup factor
 */
namespace factorization {


template <bool IsSorted, typename ValueType, typename IndexType>
void find_missing_diagonal_elements(
    const matrix::Fbcsr<ValueType, IndexType> *mtx,
    IndexType *const elements_to_add_per_row,
    bool *const changes_required) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void add_missing_diagonal_elements(
    const matrix::Fbcsr<ValueType, IndexType> *mtx, ValueType *new_values,
    IndexType *new_col_idxs,
    const IndexType *row_ptrs_addition) GKO_NOT_IMPLEMENTED;


template <typename ValueType, typename IndexType>
void add_diagonal_blocks(std::shared_ptr<const DpcppExecutor> exec,
                         matrix::Fbcsr<ValueType, IndexType> *mtx,
                         bool is_sorted) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_ADD_DIAGONAL_BLOCKS_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_row_ptrs_BLU(
    std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *system_matrix,
    IndexType *l_row_ptrs, IndexType *u_row_ptrs) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_ROW_PTRS_BLU_KERNEL);


template <typename ValueType, typename IndexType>
void initialize_BLU(
    const std::shared_ptr<const DpcppExecutor> exec,
    const matrix::Fbcsr<ValueType, IndexType> *const system_matrix,
    matrix::Fbcsr<ValueType, IndexType> *const l_factor,
    matrix::Fbcsr<ValueType, IndexType> *const u_factor) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_FACTORIZATION_INITIALIZE_BLU_KERNEL);


}  // namespace factorization
}  // namespace dpcpp
}  // namespace kernels
}  // namespace gko
