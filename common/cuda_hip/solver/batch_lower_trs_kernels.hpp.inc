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
__device__ __forceinline__ void naive_lower_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;
    const int* const __restrict__ row_ptrs = L_entry.row_ptrs;
    const int* const __restrict__ col_idxs = L_entry.col_idxs;
    const ValueType* const __restrict__ mat_values_entry = L_entry.values;

    if (threadIdx.x == 0) {
        // L * x = b
        for (int row_idx = 0; row_idx < num_rows; row_idx++) {
            ValueType sum{0};
            for (int i = row_ptrs[row_idx]; i < row_ptrs[row_idx + 1] - 1;
                 i++) {
                sum += mat_values_entry[i] * x_entry[col_idxs[i]];
            }
            x_entry[row_idx] = (b_entry[row_idx] - sum) /
                               mat_values_entry[row_ptrs[row_idx + 1] - 1];
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void naive_lower_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const auto nrows = L_entry.num_rows;
    const auto nnz_stored_per_row = L_entry.num_stored_elems_per_row;
    const auto stride = L_entry.stride;

    if (threadIdx.x == 0) {
        for (int row_idx = 0; row_idx < nrows; row_idx++) {
            ValueType sum = zero<ValueType>();
            ValueType diag_val = zero<ValueType>();
            for (int idx = 0; idx < nnz_stored_per_row; idx++) {
                const auto col_idx = L_entry.col_idxs[row_idx + idx * stride];

                if (col_idx == invalid_index<int>()) {
                    break;
                } else if (col_idx < row_idx) {
                    sum += L_entry.values[row_idx + idx * stride] *
                           x_entry[col_idx];
                } else if (col_idx == row_idx) {
                    diag_val = L_entry.values[row_idx + idx * stride];
                }
            }

            x_entry[row_idx] = (b_entry[row_idx] - sum) / diag_val;
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void naive_lower_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;

    if (threadIdx.x == 0) {
        // L * x = b
        for (int row_idx = 0; row_idx < num_rows; row_idx++) {
            ValueType sum = zero<ValueType>();
            for (int col_idx = 0; col_idx < row_idx; col_idx++) {
                sum += L_entry.values[row_idx * L_entry.stride + col_idx] *
                       x_entry[col_idx];
            }

            x_entry[row_idx] =
                (b_entry[row_idx] - sum) /
                L_entry.values[row_idx * L_entry.stride + row_idx];
        }
    }
}


template <typename ValueType>
__device__ __forceinline__ void independent_thread_scheduling_lower_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;
    const int* const __restrict__ row_ptrs = L_entry.row_ptrs;
    const int* const __restrict__ col_idxs = L_entry.col_idxs;
    const ValueType* const __restrict__ mat_values_entry = L_entry.values;

    for (int row_index = threadIdx.x; row_index < num_rows;
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
__device__ __forceinline__ void independent_thread_scheduling_lower_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const auto nrows = L_entry.num_rows;
    const auto nnz_stored_per_row = L_entry.num_stored_elems_per_row;
    const auto stride = L_entry.stride;

    for (int row_idx = threadIdx.x; row_idx < nrows;
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
__device__ __forceinline__ void independent_thread_scheduling_lower_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;

    for (int row_idx = threadIdx.x; row_idx < num_rows;
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

template <typename ValueType>
__device__ __forceinline__ void warp_synchronous_lower_trsv(
    const gko::batch_csr::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;
    const int* const __restrict__ row_ptrs = L_entry.row_ptrs;
    const int* const __restrict__ col_idxs = L_entry.col_idxs;
    const ValueType* const __restrict__ mat_values_entry = L_entry.values;

    for (int row_index = threadIdx.x; row_index < num_rows;
         row_index += static_cast<int>(blockDim.x)) {
        ValueType sum = 0;

        const int start = row_ptrs[row_index];
        const int end = row_ptrs[row_index + 1] - 1;

        int i = start;
        bool completed = false;

        while (!completed) {
            const int col_index = col_idxs[i];

            if (i < end && is_finite(load(x_entry, col_index))) {
                sum += mat_values_entry[i] * load(x_entry, col_index);
                i++;
            }

            if (i == end) {
                ValueType val =
                    (b_entry[row_index] - sum) / mat_values_entry[end];
                store(x_entry, row_index, val);
                completed = true;
            }
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void warp_synchronous_lower_trsv(
    const gko::batch_ell::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const auto nrows = L_entry.num_rows;
    const auto nnz_stored_per_row = L_entry.num_stored_elems_per_row;
    const auto stride = L_entry.stride;

    for (int row_idx = threadIdx.x; row_idx < nrows;
         row_idx += static_cast<int>(blockDim.x)) {
        ValueType sum = zero<ValueType>();
        ValueType diag_val = zero<ValueType>();

        bool completed = false;
        int idx = 0;

        while (!completed) {
            if (idx >= nnz_stored_per_row) {
                break;
            }

            const auto col_idx = L_entry.col_idxs[row_idx + idx * stride];

            if (col_idx < row_idx & is_finite(load(x_entry, col_idx))) {
                sum += L_entry.values[row_idx + idx * stride] *
                       load(x_entry, col_idx);
                idx++;
            } else if (col_idx == row_idx) {
                diag_val = L_entry.values[row_idx + idx * stride];
                ValueType val = (b_entry[row_idx] - sum) / diag_val;

                store(x_entry, row_idx, val);
                completed = true;
            }
        }
    }
}

template <typename ValueType>
__device__ __forceinline__ void warp_synchronous_lower_trsv(
    const gko::batch_dense::BatchEntry<const ValueType>& L_entry,
    const ValueType* const __restrict__ b_entry,
    ValueType* const __restrict__ x_entry)
{
    const int num_rows = L_entry.num_rows;

    for (int row_idx = threadIdx.x; row_idx < num_rows;
         row_idx += static_cast<int>(blockDim.x)) {
        ValueType sum = zero<ValueType>();

        int col_idx = 0;
        bool completed = false;

        while (!completed) {
            if (col_idx < row_idx && is_finite(load(x_entry, col_idx))) {
                sum += L_entry.values[row_idx * L_entry.stride + col_idx] *
                       load(x_entry, col_idx);
                col_idx++;
            }

            if (col_idx == row_idx) {
                ValueType val =
                    (b_entry[row_idx] - sum) /
                    L_entry.values[row_idx * L_entry.stride + row_idx];
                store(x_entry, row_idx, val);
                completed = true;
            }
        }
    }
}


template <typename BatchMatrixType, typename ValueType>
__global__ void apply_kernel(const BatchMatrixType L,
                             const ValueType* const __restrict__ b,
                             ValueType* const __restrict__ x)
{
    const auto nbatch = L.num_batch;
    const auto num_rows = L.num_rows;

    for (size_t batch_id = blockIdx.x; batch_id < nbatch;
         batch_id += gridDim.x) {
        const auto L_entry = gko::batch::batch_entry(L, batch_id);
        const ValueType* const b_entry =
            gko::batch::batch_entry_ptr(b, 1, num_rows, batch_id);
        ValueType* const x_entry =
            gko::batch::batch_entry_ptr(x, 1, num_rows, batch_id);

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__CUDA_ARCH__)
        // L * x = b
        naive_lower_trsv(L_entry, b_entry, x_entry);
        return;
#endif

        for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
            x_entry[i] = nan<ValueType>();
        }
        __syncthreads();

// busy-waiting while loop approach
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        //  Kernels for volta architecture and its successors
        //  (make use of independent thread scheduling)
        independent_thread_scheduling_lower_trsv(L_entry, b_entry, x_entry);
#else

        warp_synchronous_lower_trsv(L_entry, b_entry, x_entry);

#endif
    }
}