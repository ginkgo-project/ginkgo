// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename IndexType>
__device__ __forceinline__ void simple_apply(
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& mat,
    const ValueType* const __restrict__ b, ValueType* const __restrict__ x)
{
    const auto num_rows = mat.num_rows;
    const auto num_stored_elements_per_row = mat.num_stored_elems_per_row;
    const auto stride = mat.stride;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += val[ind] * b[col_idx];
            }
        }
        x[tidx] = temp;
    }
}

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void simple_apply_kernel(const gko::batch::matrix::
                                                      ell::uniform_batch<
                                                          const ValueType,
                                                          IndexType>
                                                          mat,
                                                  const gko::batch::
                                                      multi_vector::
                                                          uniform_batch<
                                                              const ValueType>
                                                              b,
                                                  const gko::batch::
                                                      multi_vector::
                                                          uniform_batch<
                                                              ValueType>
                                                              x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        simple_apply(mat_b, b_b.values, x_b.values);
    }
}


template <typename ValueType, typename IndexType>
__device__ __forceinline__ void advanced_apply(
    const ValueType alpha,
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& mat,
    const ValueType* const __restrict__ b, const ValueType beta,
    ValueType* const __restrict__ x)
{
    const auto num_rows = mat.num_rows;
    const auto num_stored_elements_per_row = mat.num_stored_elems_per_row;
    const auto stride = mat.stride;
    const auto val = mat.values;
    const auto col = mat.col_idxs;
    for (int tidx = threadIdx.x; tidx < num_rows; tidx += blockDim.x) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < num_stored_elements_per_row; idx++) {
            const auto ind = tidx + idx * stride;
            const auto col_idx = col[ind];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += alpha * val[ind] * b[col_idx];
            }
        }
        x[tidx] = temp + beta * x[tidx];
    }
}

template <typename ValueType, typename IndexType>
__global__ __launch_bounds__(
    default_block_size,
    sm_oversubscription) void advanced_apply_kernel(const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                alpha,
                                                    const gko::batch::matrix::
                                                        ell::uniform_batch<
                                                            const ValueType,
                                                            IndexType>
                                                            mat,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                b,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                const ValueType>
                                                                beta,
                                                    const gko::batch::
                                                        multi_vector::
                                                            uniform_batch<
                                                                ValueType>
                                                                x)
{
    for (size_type batch_id = blockIdx.x; batch_id < mat.num_batch_items;
         batch_id += gridDim.x) {
        const auto mat_b =
            gko::batch::matrix::extract_batch_item(mat, batch_id);
        const auto b_b = gko::batch::extract_batch_item(b, batch_id);
        const auto x_b = gko::batch::extract_batch_item(x, batch_id);
        const auto alpha_b = gko::batch::extract_batch_item(alpha, batch_id);
        const auto beta_b = gko::batch::extract_batch_item(beta, batch_id);
        advanced_apply(alpha_b.values[0], mat_b, b_b.values, beta_b.values[0],
                       x_b.values);
    }
}
