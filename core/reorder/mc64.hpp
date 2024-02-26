// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GKO_CORE_REORDER_MC64_HPP_
#define GKO_CORE_REORDER_MC64_HPP_


#include <ginkgo/core/reorder/mc64.hpp>


#include <ginkgo/core/base/array.hpp>
#include <ginkgo/core/matrix/csr.hpp>


#include "core/components/addressable_pq.hpp"


namespace gko {
namespace experimental {
namespace reorder {
namespace mc64 {


#define GKO_DECLARE_MC64_INITIALIZE_WEIGHTS(ValueType, IndexType) \
    void initialize_weights(                                      \
        const matrix::Csr<ValueType, IndexType>* mtx,             \
        array<remove_complex<ValueType>>& weights_array,          \
        array<remove_complex<ValueType>>& dual_u_array,           \
        array<remove_complex<ValueType>>& row_maxima_array,       \
        gko::experimental::reorder::mc64_strategy strategy)

#define GKO_DECLARE_MC64_INITIAL_MATCHING(ValueType, IndexType)              \
    void initial_matching(                                                   \
        size_type num_rows, const IndexType* row_ptrs,                       \
        const IndexType* col_idxs, const array<ValueType>& weights_array,    \
        const array<ValueType>& dual_u_array, array<IndexType>& permutation, \
        array<IndexType>& inv_permutation,                                   \
        array<IndexType>& matched_idxs_array,                                \
        array<IndexType>& unmatched_rows_array, ValueType tolerance)

#define GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH(ValueType, IndexType)   \
    void shortest_augmenting_path(                                        \
        size_type num_rows, const IndexType* row_ptrs,                    \
        const IndexType* col_idxs, array<ValueType>& weights_array,       \
        array<ValueType>& dual_u_array, array<ValueType>& distance_array, \
        array<IndexType>& permutation, array<IndexType>& inv_permutation, \
        IndexType root, array<IndexType>& parents_array,                  \
        array<IndexType>& generation_array,                               \
        array<IndexType>& marked_cols_array,                              \
        array<IndexType>& matched_idxs_array,                             \
        addressable_priority_queue<ValueType, IndexType>& queue,          \
        std::vector<IndexType>& q_j, ValueType tolerance)

#define GKO_DECLARE_MC64_COMPUTE_SCALING(ValueType, IndexType)              \
    void compute_scaling(                                                   \
        const matrix::Csr<ValueType, IndexType>* mtx,                       \
        const array<remove_complex<ValueType>>& weights_array,              \
        const array<remove_complex<ValueType>>& dual_u_array,               \
        const array<remove_complex<ValueType>>& row_maxima_array,           \
        const array<IndexType>& permutation,                                \
        const array<IndexType>& matched_idxs_array, mc64_strategy strategy, \
        ValueType* row_scaling, ValueType* col_scaling)


template <typename ValueType, typename IndexType>
GKO_DECLARE_MC64_INITIALIZE_WEIGHTS(ValueType, IndexType);

template <typename ValueType, typename IndexType>
GKO_DECLARE_MC64_INITIAL_MATCHING(ValueType, IndexType);

template <typename ValueType, typename IndexType>
GKO_DECLARE_MC64_SHORTEST_AUGMENTING_PATH(ValueType, IndexType);

template <typename ValueType, typename IndexType>
GKO_DECLARE_MC64_COMPUTE_SCALING(ValueType, IndexType);


}  // namespace mc64
}  // namespace reorder
}  // namespace experimental
}  // namespace gko

#endif  // GKO_CORE_REORDER_MC64_HPP_
