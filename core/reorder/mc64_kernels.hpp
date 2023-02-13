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

#ifndef GKO_CORE_REORDER_MC64_KERNELS_HPP_
#define GKO_CORE_REORDER_MC64_KERNELS_HPP_


#include <ginkgo/core/reorder/mc64.hpp>


#include <list>
#include <memory>


#include <ginkgo/core/base/executor.hpp>
#include <ginkgo/core/base/types.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/base/kernel_declaration.hpp"
#include "core/components/addressable_pq.hpp"


namespace gko {
namespace kernels {


#define GKO_DECLARE_MC64_INITIALIZE_WEIGHTS_KERNEL(ValueType, IndexType) \
    void initialize_weights(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                     \
        const matrix::Csr<ValueType, IndexType>* mtx,                    \
        array<remove_complex<ValueType>>& weights_array,                 \
        array<remove_complex<ValueType>>& dual_u_array,                  \
        array<remove_complex<ValueType>>& distance_array,                \
        array<remove_complex<ValueType>>& row_maxima_array,              \
        gko::reorder::mc64_strategy strategy)


#define GKO_DECLARE_MC64_INITIAL_MATCHING_KERNEL(ValueType, IndexType)       \
    void initial_matching(                                                   \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,     \
        const IndexType* row_ptrs, const IndexType* col_idxs,                \
        const array<ValueType>& weights_array,                               \
        const array<ValueType>& dual_u_array, array<IndexType>& permutation, \
        array<IndexType>& inv_permutation,                                   \
        array<IndexType>& matched_idxs_array,                                \
        array<IndexType>& unmatched_rows_array, ValueType tolerance)


#define GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH_KERNEL(ValueType, IndexType) \
    void shortest_augmenting_path(                                             \
        std::shared_ptr<const DefaultExecutor> exec, size_type num_rows,       \
        const IndexType* row_ptrs, const IndexType* col_idxs,                  \
        array<ValueType>& weights_array, array<ValueType>& dual_u_array,       \
        array<ValueType>& distance_array, array<IndexType>& permutation,       \
        array<IndexType>& inv_permutation, IndexType root,                     \
        array<IndexType>& parents_array, array<IndexType>& handles_array,      \
        array<IndexType>& generation_array,                                    \
        array<IndexType>& marked_cols_array,                                   \
        array<IndexType>& matched_idxs_array,                                  \
        addressable_priority_queue<ValueType, IndexType>& Q,                   \
        std::vector<IndexType>& q_j, ValueType tolerance)


#define GKO_DECLARE_MC64_COMPUTE_SCALING_KERNEL(ValueType, IndexType) \
    void compute_scaling(                                             \
        std::shared_ptr<const DefaultExecutor> exec,                  \
        const matrix::Csr<ValueType, IndexType>* mtx,                 \
        const array<remove_complex<ValueType>>& weights_array,        \
        const array<remove_complex<ValueType>>& dual_u_array,         \
        const array<remove_complex<ValueType>>& row_maxima_array,     \
        const array<IndexType>& permutation,                          \
        const array<IndexType>& matched_idxs_array,                   \
        gko::reorder::mc64_strategy strategy,                         \
        gko::matrix::Diagonal<ValueType>* row_scaling,                \
        gko::matrix::Diagonal<ValueType>* col_scaling)


#define GKO_DECLARE_ALL_AS_TEMPLATES                                        \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_MC64_INITIALIZE_WEIGHTS_KERNEL(ValueType, IndexType);       \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_MC64_INITIAL_MATCHING_KERNEL(ValueType, IndexType);         \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH_KERNEL(ValueType, IndexType); \
    template <typename ValueType, typename IndexType>                       \
    GKO_DECLARE_MC64_COMPUTE_SCALING_KERNEL(ValueType, IndexType)


GKO_DECLARE_FOR_ALL_EXECUTOR_NAMESPACES(mc64, GKO_DECLARE_ALL_AS_TEMPLATES);


#undef GKO_DECLARE_ALL_AS_TEMPLATES


}  // namespace kernels
}  // namespace gko


#endif  // GKO_CORE_REORDER_MC64_KERNELS_HPP_
