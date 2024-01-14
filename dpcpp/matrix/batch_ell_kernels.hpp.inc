// SPDX-FileCopyrightText: 2017 - 2024 The Ginkgo authors
//
// SPDX-License-Identifier: BSD-3-Clause

template <typename ValueType, typename IndexType>
__dpct_inline__ void simple_apply_kernel(
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& mat,
    const ValueType* b, ValueType* x, sycl::nd_item<3>& item_ct1)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < mat.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
            const auto col_idx = mat.col_idxs[tidx + idx * mat.stride];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += mat.values[tidx + idx * mat.stride] * b[col_idx];
            }
        }
        x[tidx] = temp;
    }
}


template <typename ValueType, typename IndexType>
__dpct_inline__ void advanced_apply_kernel(
    const ValueType alpha,
    const gko::batch::matrix::ell::batch_item<const ValueType, IndexType>& mat,
    const ValueType* b, const ValueType beta, ValueType* x,
    sycl::nd_item<3>& item_ct1)
{
    for (int tidx = item_ct1.get_local_linear_id(); tidx < mat.num_rows;
         tidx += item_ct1.get_local_range().size()) {
        auto temp = zero<ValueType>();
        for (size_type idx = 0; idx < mat.num_stored_elems_per_row; idx++) {
            const auto col_idx = mat.col_idxs[tidx + idx * mat.stride];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                temp += mat.values[tidx + idx * mat.stride] * b[col_idx];
            }
        }
        x[tidx] = alpha * temp + beta * x[tidx];
    }
}


template <typename ValueType, typename IndexType>
__dpct_inline__ void scale_kernel(
    const ValueType* const col_scale, const ValueType* const row_scale,
    gko::batch::matrix::ell::batch_item<ValueType, IndexType>& mat,
    sycl::nd_item<3>& item_ct1)
{
    for (int row = item_ct1.get_local_linear_id(); row < mat.num_rows;
         row += item_ct1.get_local_range().size()) {
        const ValueType row_scalar = row_scale[row];
        for (auto k = 0; k < mat.num_stored_elems_per_row; k++) {
            auto col_idx = mat.col_idxs[row + mat.stride * k];
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                mat.values[row + mat.stride * k] *=
                    row_scalar * col_scale[col_idx];
            }
        }
    }
}


template <typename ValueType, typename IndexType>
__dpct_inline__ void add_scaled_identity_kernel(
    const ValueType alpha, const ValueType beta,
    const gko::batch::matrix::ell::batch_item<ValueType, IndexType>& mat,
    sycl::nd_item<3>& item_ct1)
{
    for (int row = item_ct1.get_local_linear_id(); row < mat.num_rows;
         row += item_ct1.get_local_range().size()) {
        for (auto k = 0; k < mat.num_stored_elems_per_row; k++) {
            auto col_idx = mat.col_idxs[row + mat.stride * k];
            mat.values[row + k * mat.stride] *= beta;
            if (col_idx == invalid_index<IndexType>()) {
                break;
            } else {
                if (row == col_idx) {
                    mat.values[row + k * mat.stride] += alpha;
                }
            }
        }
    }
}
