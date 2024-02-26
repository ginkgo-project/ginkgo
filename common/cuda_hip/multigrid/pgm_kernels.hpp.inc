// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename IndexType>
void sort_agg(std::shared_ptr<const DefaultExecutor> exec, IndexType num,
              IndexType* row_idxs, IndexType* col_idxs)
{
    auto it = thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));
    thrust::sort(thrust_policy(exec), it, it + num);
}

GKO_INSTANTIATE_FOR_EACH_INDEX_TYPE(GKO_DECLARE_PGM_SORT_AGG_KERNEL);


template <typename ValueType, typename IndexType>
void sort_row_major(std::shared_ptr<const DefaultExecutor> exec, size_type nnz,
                    IndexType* row_idxs, IndexType* col_idxs, ValueType* vals)
{
    auto vals_it = as_device_type(vals);
    auto it = thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));
    // Because reduce_by_key is not deterministic, so we do not need
    // stable_sort_by_key
    // TODO: If we have deterministic reduce_by_key, it should be
    // stable_sort_by_key
    thrust::sort_by_key(thrust_policy(exec), it, it + nnz, vals_it);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(GKO_DECLARE_PGM_SORT_ROW_MAJOR);


template <typename ValueType, typename IndexType>
void compute_coarse_coo(std::shared_ptr<const DefaultExecutor> exec,
                        size_type fine_nnz, const IndexType* row_idxs,
                        const IndexType* col_idxs, const ValueType* vals,
                        matrix::Coo<ValueType, IndexType>* coarse_coo)
{
    auto vals_it = as_device_type(vals);
    auto key_it =
        thrust::make_zip_iterator(thrust::make_tuple(row_idxs, col_idxs));

    auto coarse_vals_it = as_device_type(coarse_coo->get_values());
    auto coarse_key_it = thrust::make_zip_iterator(thrust::make_tuple(
        coarse_coo->get_row_idxs(), coarse_coo->get_col_idxs()));

    thrust::reduce_by_key(thrust_policy(exec), key_it, key_it + fine_nnz,
                          vals_it, coarse_key_it, coarse_vals_it);
}

GKO_INSTANTIATE_FOR_EACH_VALUE_AND_INDEX_TYPE(
    GKO_DECLARE_PGM_COMPUTE_COARSE_COO);
