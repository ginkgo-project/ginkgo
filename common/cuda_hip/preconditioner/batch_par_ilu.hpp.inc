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


/**
 * Batch parilu0 preconditioner class.
 *
 */
template <typename ValueType>
class batch_parilu0 final {
public:
    using value_type = ValueType;

    /**
     * @param l_batch  Lower triangular factor that was externally generated.
     * @param u_batch  Upper triangular factor that was externally generated.
     * @param is_fallback_required indicates what trsv is to be used
     */
    batch_parilu0(const gko::batch_csr::UniformBatch<const ValueType>& l_batch,
                  const gko::batch_csr::UniformBatch<const ValueType>& u_batch,
                  const bool& is_fallback_required)
        : l_batch_{l_batch},
          u_batch_{u_batch},
          is_fallback_required_{is_fallback_required}
    {}

    /**
     * The size of the work vector required per batch entry. (takes into account
     * both- generation and application)
     */
    __host__ __device__ static constexpr int dynamic_work_size(int nrows,
                                                               int nnz)
    {
        return nrows;
    }

    /**
     * Complete the precond generation process.
     *
     * @param mat  Matrix for which to build an ILU-type preconditioner.
     */
    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_csr::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id, const gko::batch_ell::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }

    __device__ __forceinline__ void generate(
        size_type batch_id,
        const gko::batch_dense::BatchEntry<const ValueType>&,
        ValueType* const __restrict__ work)
    {
        l_entry_ = gko::batch::batch_entry(l_batch_, batch_id);
        u_entry_ = gko::batch::batch_entry(u_batch_, batch_id);
        work_ = work;
    }


    // TODO: (later) Move trsv functions to a batch_trsv_kernels.hpp.inc file
    // (in case we have a separate batched lower and upper traingular solver
    // (with public, core stuff etc.))
    __device__ __forceinline__ void naive_lower_trsv(
        const ValueType* const __restrict__ r) const
    {
        const int num_rows = l_entry_.num_rows;
        const int* const __restrict__ L_row_ptrs = l_entry_.row_ptrs;
        const int* const __restrict__ L_col_idxs = l_entry_.col_idxs;
        const ValueType* const __restrict__ L_values = l_entry_.values;

        if (threadIdx.x == 0) {
            for (int row_idx = 0; row_idx < num_rows; row_idx++) {
                const int row_st = L_row_ptrs[row_idx];
                const int row_end = L_row_ptrs[row_idx + 1];
                ValueType sum{0};
                for (int i = row_st; i < row_end - 1; i++) {
                    const int col_idx = L_col_idxs[i];
                    sum += L_values[i] * work_[col_idx];
                }
                work_[row_idx] = (r[row_idx] - sum) / L_values[row_end - 1];
            }
        }
    }

    __device__ __forceinline__ void naive_upper_trsv(
        ValueType* const __restrict__ z) const
    {
        const int num_rows = u_entry_.num_rows;
        const int* const __restrict__ U_row_ptrs = u_entry_.row_ptrs;
        const int* const __restrict__ U_col_idxs = u_entry_.col_idxs;
        const ValueType* const __restrict__ U_values = u_entry_.values;

        if (threadIdx.x == 0) {
            for (int row_idx = num_rows - 1; row_idx >= 0; row_idx--) {
                const int row_st = U_row_ptrs[row_idx];
                const int row_end = U_row_ptrs[row_idx + 1];
                ValueType sum{0};
                for (int i = row_end - 1; i > row_st; i--) {
                    const int col_idx = U_col_idxs[i];
                    sum += U_values[i] * z[col_idx];
                }
                z[row_idx] = (work_[row_idx] - sum) / U_values[row_st];
            }
        }
    }

    __device__ __forceinline__ void warp_synchronous_lower_trsv(
        const ValueType* const __restrict__ r) const
    {
        const int num_rows = l_entry_.num_rows;
        const int* const __restrict__ L_row_ptrs = l_entry_.row_ptrs;
        const int* const __restrict__ L_col_idxs = l_entry_.col_idxs;
        const ValueType* const __restrict__ L_values = l_entry_.values;

        const int row_index = threadIdx.x;

        if (row_index < num_rows) {
            ValueType sum = zero<ValueType>();

            const int start = L_row_ptrs[row_index];
            const int end = L_row_ptrs[row_index + 1] - 1;
            int i = start;

            bool completed = false;

            while (!completed) {
                const int col_index = L_col_idxs[i];
                if (i < end && is_finite(load(work_, col_index))) {
                    sum += L_values[i] * load(work_, col_index);
                    i++;
                }

                if (i == end) {
                    ValueType val = (r[row_index] - sum) / L_values[end];
                    store(work_, row_index, val);
                    completed = true;
                }
            }
        }
    }

    __device__ __forceinline__ void warp_synchronous_upper_trsv(
        ValueType* const __restrict__ z) const
    {
        const int num_rows = u_entry_.num_rows;
        const int* const __restrict__ U_row_ptrs = u_entry_.row_ptrs;
        const int* const __restrict__ U_col_idxs = u_entry_.col_idxs;
        const ValueType* const __restrict__ U_values = u_entry_.values;

        const int row_index = num_rows - 1 - threadIdx.x;

        if (row_index >= 0) {
            ValueType sum = zero<ValueType>();

            const int start = U_row_ptrs[row_index];
            const int end = U_row_ptrs[row_index + 1] - 1;
            int i = end;

            bool completed = false;

            while (!completed) {
                const int col_index = U_col_idxs[i];

                if (i > start && is_finite(load(z, col_index))) {
                    sum += U_values[i] * load(z, col_index);
                    i--;
                }

                if (i == start) {
                    ValueType val = (work_[row_index] - sum) / U_values[start];
                    store(z, row_index, val);
                    completed = true;
                }
            }
        }
    }

    __device__ __forceinline__ void independent_thread_scheduling_lower_trsv(
        const ValueType* const __restrict__ r) const
    {
        const int num_rows = l_entry_.num_rows;
        const int* const __restrict__ L_row_ptrs = l_entry_.row_ptrs;
        const int* const __restrict__ L_col_idxs = l_entry_.col_idxs;
        const ValueType* const __restrict__ L_values = l_entry_.values;

        const int row_index = threadIdx.x;

        if (row_index < num_rows) {
            ValueType sum = zero<ValueType>();

            const int start = L_row_ptrs[row_index];
            const int end = L_row_ptrs[row_index + 1] - 1;

            for (int i = start; i < end; i++) {
                const int col_index = L_col_idxs[i];
                while (!is_finite(load(work_, col_index))) {
                }
                sum += L_values[i] * load(work_, col_index);
            }

            ValueType val = (r[row_index] - sum) / L_values[end];
            store(work_, row_index, val);
        }
    }

    __device__ __forceinline__ void independent_thread_scheduling_upper_trsv(
        ValueType* const __restrict__ z) const
    {
        const int num_rows = u_entry_.num_rows;
        const int* const __restrict__ U_row_ptrs = u_entry_.row_ptrs;
        const int* const __restrict__ U_col_idxs = u_entry_.col_idxs;
        const ValueType* const __restrict__ U_values = u_entry_.values;

        const int row_index = num_rows - 1 - threadIdx.x;

        if (row_index >= 0) {
            ValueType sum = zero<ValueType>();
            const int start = U_row_ptrs[row_index];
            const int end = U_row_ptrs[row_index + 1] - 1;

            for (int i = end; i > start; i--) {
                const int col_index = U_col_idxs[i];
                while (!is_finite(load(z, col_index))) {
                }
                sum += U_values[i] * load(z, col_index);
            }

            ValueType val = (work_[row_index] - sum) / U_values[start];
            store(z, row_index, val);
        }
    }


    __device__ __forceinline__ void apply(const int num_rows,
                                          const ValueType* const __restrict__ r,
                                          ValueType* const __restrict__ z) const
    {
        // z = precond * r i.e.  L * U * z = r
        // sparse triangular solves
        // L * work_ = r  (find work_) Note: L is in sorted order
        // U * z = work_  (then find z)  Note: U is in sorted order

        // naive_lower_trsv(r);
        // naive_upper_trsv(z);

        for (int i = threadIdx.x; i < num_rows; i += blockDim.x) {
            store(work_, i, nan<ValueType>());
            store(z, i, nan<ValueType>());
        }
        __syncthreads();

        // busy-waiting while loop approach
        if (is_fallback_required_) {
            warp_synchronous_lower_trsv(r);
            __syncthreads();
            warp_synchronous_upper_trsv(z);
        } else {
            // kernels for volta architecture and its successors (which make use
            // of independent thread scheduling (threads in a warp do not
            // necessarily run synchronously))
            independent_thread_scheduling_lower_trsv(r);
            __syncthreads();
            independent_thread_scheduling_upper_trsv(z);
        }
    }


private:
    const gko::batch_csr::UniformBatch<const value_type> l_batch_;
    const gko::batch_csr::UniformBatch<const value_type> u_batch_;
    gko::batch_csr::BatchEntry<const value_type> l_entry_;
    gko::batch_csr::BatchEntry<const value_type> u_entry_;
    ValueType* __restrict__ work_;
    const bool is_fallback_required_;
};