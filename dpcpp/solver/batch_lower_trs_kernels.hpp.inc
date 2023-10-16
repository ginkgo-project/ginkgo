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

template <typename ValueType>
__dpct_inline__ void naive_lower_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;
    const int* const __restrict__ row_ptrs = L_entry.row_ptrs;
    const int* const __restrict__ col_idxs = L_entry.col_idxs;
    const ValueType* const __restrict__ mat_values_entry = L_entry.values;

    // L * x = b
    for (int row_idx = 0; row_idx < num_rows; row_idx++) {
        ValueType sum{0};
        for (int i = row_ptrs[row_idx]; i < row_ptrs[row_idx + 1] - 1; i++) {
            sum += mat_values_entry[i] * x_entry[col_idxs[i]];
        }
        x_entry[row_idx] = (b_entry[row_idx] - sum) /
                           mat_values_entry[row_ptrs[row_idx + 1] - 1];
    }
}

template <typename ValueType>
__dpct_inline__ void naive_lower_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const auto nrows = L_entry.num_rows;
    const auto nnz_stored_per_row = L_entry.num_stored_elems_per_row;
    const auto stride = L_entry.stride;

    for (int row_idx = 0; row_idx < nrows; row_idx++) {
        ValueType sum = zero<ValueType>();
        ValueType diag_val = zero<ValueType>();
        for (int idx = 0; idx < nnz_stored_per_row; idx++) {
            const auto col_idx = L_entry.col_idxs[row_idx + idx * stride];

            if (col_idx == invalid_index<int>()) {
                break;
            } else if (col_idx < row_idx) {
                sum +=
                    L_entry.values[row_idx + idx * stride] * x_entry[col_idx];
            } else if (col_idx == row_idx) {
                diag_val = L_entry.values[row_idx + idx * stride];
            }
        }

        x_entry[row_idx] = (b_entry[row_idx] - sum) / diag_val;
    }
}

template <typename ValueType>
__dpct_inline__ void naive_lower_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;

    // L * x = b
    for (int row_idx = 0; row_idx < num_rows; row_idx++) {
        ValueType sum = zero<ValueType>();
        for (int col_idx = 0; col_idx < row_idx; col_idx++) {
            sum += L_entry.values[row_idx * L_entry.stride + col_idx] *
                   x_entry[col_idx];
        }

        x_entry[row_idx] = (b_entry[row_idx] - sum) /
                           L_entry.values[row_idx * L_entry.stride + row_idx];
    }
}

/*
template <typename ValueType>
__dpct_inline__ void independent_thread_scheduling_lower_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry,
    sycl::nd_item<3> item_ct1)
{
    const int num_rows = L_entry.num_rows;
    const int* const __restrict__ row_ptrs = L_entry.row_ptrs;
    const int* const __restrict__ col_idxs = L_entry.col_idxs;
    const ValueType* const __restrict__ mat_values_entry = L_entry.values;

    for (int row_index = item_ct1.get_local_linear_id(); row_index < num_rows;
         row_index += static_cast<int>(blockDim.x)) {
        ValueType sum = 0;
        const int start = row_ptrs[row_index];
        const int end = row_ptrs[row_index + 1] - 1;

        for (int i = start; i < end; i++) {
            const int col_index = col_idxs[i];
            while (!is_finite(load(x_entry, col_index))) {
            }
            sum += mat_values_entry[i] * load(x_entry, col_index);
        }
        ValueType val = (b_entry[row_index] - sum) / mat_values_entry[end];
        store(x_entry, row_index, val);
    }
}

template <typename ValueType>
__dpct_inline__ void independent_thread_scheduling_lower_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry,
    sycl::nd_item<3> item_ct1)
{
    const auto nrows = L_entry.num_rows;
    const auto nnz_stored_per_row = L_entry.num_stored_elems_per_row;
    const auto stride = L_entry.stride;

    for (int row_idx = item_ct1.get_local_linear_id(); row_idx < nrows;
         row_idx += static_cast<int>(blockDim.x)) {
        ValueType sum = zero<ValueType>();
        ValueType diag_val = zero<ValueType>();

        for (int idx = 0; idx < nnz_stored_per_row; idx++) {
            const auto col_idx = L_entry.col_idxs[row_idx + idx * stride];

            if (col_idx == invalid_index<int>()) {
                break;
            } else if (col_idx < row_idx) {
                while (!is_finite(load(x_entry, col_idx))) {
                }
                sum += L_entry.values[row_idx + idx * stride] *
                       load(x_entry, col_idx);
            } else if (col_idx == row_idx) {
                diag_val = L_entry.values[row_idx + idx * stride];
            }
        }

        ValueType val = (b_entry[row_idx] - sum) / diag_val;
        store(x_entry, row_idx, val);
    }
}

template <typename ValueType>
__dpct_inline__ void independent_thread_scheduling_lower_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry,
    sycl::nd_item<3> item_ct1)
{
    const int num_rows = L_entry.num_rows;

    for (int row_idx = item_ct1.get_local_linear_id(); row_idx < num_rows;
         row_idx += static_cast<int>(blockDim.x)) {
        ValueType sum = zero<ValueType>();

        for (int col_idx = 0; col_idx < row_idx; col_idx++) {
            while (!is_finite(load(x_entry, col_idx))) {
            }

            sum += L_entry.values[row_idx * L_entry.stride + col_idx] *
                   load(x_entry, col_idx);
        }

        ValueType val = (b_entry[row_idx] - sum) /
                        L_entry.values[row_idx * L_entry.stride + row_idx];
        store(x_entry, row_idx, val);
    }
}
*/
template <typename ValueType>
__dpct_inline__ void warp_synchronous_lower_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry, sycl::nd_item<3> item_ct1)
{
    const int num_rows = L_entry.num_rows;
    const int* const __restrict__ row_ptrs = L_entry.row_ptrs;
    const int* const __restrict__ col_idxs = L_entry.col_idxs;
    const ValueType* const __restrict__ mat_values_entry = L_entry.values;

    for (int row_index = item_ct1.get_local_linear_id(); row_index < num_rows;
         row_index += item_ct1.get_local_range().size()) {
        ValueType sum = 0;

        const int start = row_ptrs[row_index];
        const int end = row_ptrs[row_index + 1] - 1;

        int i = start;
        bool completed = false;

        while (!completed) {
            const int col_index = col_idxs[i];

            if (i < end && is_finite(x_entry[col_index])) {
                sum += mat_values_entry[i] * x_entry[col_index];
                i++;
            }

            if (i == end) {
                ValueType val =
                    (b_entry[row_index] - sum) / mat_values_entry[end];
                x_entry[row_index] = val;
                completed = true;
            }
        }
    }
}

template <typename ValueType>
__dpct_inline__ void warp_synchronous_lower_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry, sycl::nd_item<3> item_ct1)
{
    const auto nrows = L_entry.num_rows;
    const auto nnz_stored_per_row = L_entry.num_stored_elems_per_row;
    const auto stride = L_entry.stride;

    for (int row_idx = item_ct1.get_local_linear_id(); row_idx < nrows;
         row_idx += item_ct1.get_local_range().size()) {
        ValueType sum = zero<ValueType>();
        ValueType diag_val = zero<ValueType>();

        bool completed = false;
        int idx = 0;

        while (!completed) {
            if (idx >= nnz_stored_per_row) {
                break;
            }

            const auto col_idx = L_entry.col_idxs[row_idx + idx * stride];

            if (col_idx < row_idx & is_finite(x_entry[col_idx])) {
                sum +=
                    L_entry.values[row_idx + idx * stride] * x_entry[col_idx];
                idx++;
            } else if (col_idx == row_idx) {
                diag_val = L_entry.values[row_idx + idx * stride];
                ValueType val = (b_entry[row_idx] - sum) / diag_val;

                x_entry[row_idx] = val;
                completed = true;
            }
        }
    }
}

template <typename ValueType>
__dpct_inline__ void warp_synchronous_lower_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry, sycl::nd_item<3> item_ct1)
{
    const int num_rows = L_entry.num_rows;

    for (int row_idx = item_ct1.get_local_linear_id(); row_idx < num_rows;
         row_idx += item_ct1.get_local_range().size()) {
        ValueType sum = zero<ValueType>();

        int col_idx = 0;
        bool completed = false;

        while (!completed) {
            if (col_idx < row_idx && is_finite(x_entry[col_idx])) {
                sum += L_entry.values[row_idx * L_entry.stride + col_idx] *
                       x_entry[col_idx];
                col_idx++;
            }

            if (col_idx == row_idx) {
                ValueType val =
                    (b_entry[row_idx] - sum) /
                    L_entry.values[row_idx * L_entry.stride + row_idx];
                x_entry[row_idx] = val;
                completed = true;
            }
        }
    }
}


template <typename BatchMatrixType, typename ValueType>
__dpct_inline__ void apply_kernel(const BatchMatrixType L,
                                  const ValueType* const __restrict__ b,
                                  ValueType* const __restrict__ x,
                                  const size_type num_rows,
                                  ValueType* __restrict__ slm_values,
                                  sycl::nd_item<3> item_ct1)
{
    // L * x = b
    if (item_ct1.get_local_linear_id() == 0) naive_lower_trsv(L, b, x);
    item_ct1.barrier(sycl::access::fence_space::local_space);

    // TODO: rewrite warp_synchronous_lower_trsv without the while loop
    /*
      for (int i = item_ct1.get_local_linear_id(); i < num_rows; i +=
      item_ct1.get_local_range().size()) { x[i] = nan<ValueType>();
      }
      item_ct1.barrier(sycl::access::fence_space::local_space);

      warp_synchronous_lower_trsv(L, b, x, item_ct1);
      item_ct1.barrier(sycl::access::fence_space::local_space);
      */
}
