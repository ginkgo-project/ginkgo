// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename IndexType>
void get_permutation(
    std::shared_ptr<const DefaultExecutor> exec, const IndexType num_vertices,
    const IndexType* const row_ptrs, const IndexType* const col_idxs,
    IndexType* const permutation, IndexType* const inv_permutation,
    const gko::reorder::starting_strategy strategy) GKO_NOT_IMPLEMENTED;

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_RCM_GET_PERMUTATION_KERNEL);